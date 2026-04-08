import argparse
import importlib.util
import json
import os
import random
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_ROOT = REPO_ROOT / "detection" / "denseclip"
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

if importlib.util.find_spec("ftfy") is None:
    if os.environ.get("DENSECLIP_ALLOW_MISSING_FTFY", "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "ftfy is required for prompt-text experiments. Install `ftfy` or set "
            "`DENSECLIP_ALLOW_MISSING_FTFY=1` to run in explicit degraded mode."
        )
    ftfy_stub = types.ModuleType("ftfy")
    ftfy_stub.fix_text = lambda text: text
    sys.modules["ftfy"] = ftfy_stub
    warnings.warn(
        "ftfy is unavailable; prompt-text token cleanup is running in explicit degraded mode.",
        RuntimeWarning,
        stacklevel=1,
    )

from untils import tokenize

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.data import ImageTransform
from fewshot.stage_a1 import binary_auroc
from run_stage3_head import load_records, resolve_device, split_records, write_csv


CONTROL_CATEGORY = "bottle"
FEATURE_SOURCE_CLIP_GLOBAL = "clip_global"
FEATURE_SOURCE_DENSECLIP_GLOBAL = "denseclip_global"
FEATURE_SOURCE_LAYER3_GAP = "layer3_gap"
FEATURE_SOURCE_CHOICES = (
    FEATURE_SOURCE_CLIP_GLOBAL,
    FEATURE_SOURCE_DENSECLIP_GLOBAL,
    FEATURE_SOURCE_LAYER3_GAP,
)
DEFAULT_NORMAL_TEMPLATES = (
    "a photo of a normal {category}.",
    "a photo of a good {category}.",
    "a photo of a flawless {category}.",
)
DEFAULT_DEFECT_TEMPLATES = (
    "a photo of a {category} with {defect}.",
    "a defective {category} showing {defect}.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-text v2 with multisplit selection and frozen visual sources.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_p_v2_multisplit_visual",
    )
    parser.add_argument("--scope", default="")
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--denseclip-image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument(
        "--feature-sources",
        default="clip_global,denseclip_global,layer3_gap",
        help="Comma-separated frozen visual sources to evaluate.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--p-l2", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_feature_sources(value: str) -> list[str]:
    sources = [token.strip() for token in value.split(",") if token.strip()]
    if not sources:
        raise ValueError("At least one feature source is required.")
    unknown = [source for source in sources if source not in FEATURE_SOURCE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown feature sources: {unknown}")
    return list(dict.fromkeys(sources))


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.transform = ImageTransform(image_size=image_size, augment=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        return {"path": path, "image": self.transform(Path(path))}


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def split_support_holdout(
    train_records: list[dict[str, object]],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.RandomState(seed)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in train_records:
        grouped[(str(record["category"]), int(record["label"]))].append(record)

    support_train: list[dict[str, object]] = []
    support_holdout: list[dict[str, object]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: str(item["path"]))
        if len(items) <= 1:
            support_train.extend(items)
            continue
        holdout_count = int(round(len(items) * holdout_fraction))
        holdout_count = max(1, holdout_count)
        holdout_count = min(len(items) - 1, holdout_count)
        holdout_indices = set(rng.permutation(len(items))[:holdout_count].tolist())
        for idx, item in enumerate(items):
            if idx in holdout_indices:
                support_holdout.append(item)
            else:
                support_train.append(item)
    if not support_train or not support_holdout:
        raise ValueError("Support holdout split produced an empty train or holdout partition.")
    return support_train, support_holdout


def build_resplits(
    train_records: list[dict[str, object]],
    holdout_fraction: float,
    seed: int,
    num_resplits: int,
) -> list[dict[str, object]]:
    resplits: list[dict[str, object]] = []
    for split_index in range(num_resplits):
        split_seed = seed + split_index
        support_train, holdout = split_support_holdout(
            train_records=train_records,
            holdout_fraction=holdout_fraction,
            seed=split_seed,
        )
        resplits.append(
            {
                "split_index": split_index,
                "split_seed": split_seed,
                "support_train": support_train,
                "holdout": holdout,
            }
        )
    return resplits


def sanitize_defect_name(defect_type: str) -> str:
    return defect_type.replace("_", " ").replace("-", " ").strip()


def build_prompt_bank(records: list[dict[str, object]]) -> dict[str, dict[str, list[str]]]:
    defect_types_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        category = str(record["category"])
        defect_type = str(record.get("defect_type") or "").strip()
        if defect_type and defect_type != "good":
            defect_types_by_category[category].add(defect_type)

    prompt_bank: dict[str, dict[str, list[str]]] = {}
    for category in sorted({str(record["category"]) for record in records}):
        normal_prompts = [template.format(category=category) for template in DEFAULT_NORMAL_TEMPLATES]
        defect_prompts: list[str] = []
        for defect_type in sorted(defect_types_by_category.get(category, set())):
            defect_name = sanitize_defect_name(defect_type)
            for template in DEFAULT_DEFECT_TEMPLATES:
                defect_prompts.append(template.format(category=category, defect=defect_name))
        if not defect_prompts:
            defect_prompts = [f"a defective {category}."]
        prompt_bank[category] = {"normal": normal_prompts, "defect": defect_prompts}
    return prompt_bank


@torch.no_grad()
def encode_clip_global_features(
    model: torch.jit.ScriptModule,
    paths: list[str],
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = ImagePathDataset(paths=paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    features: dict[str, torch.Tensor] = {}
    for batch in loader:
        images = batch["image"].to(device)
        image_features = F.normalize(model.encode_image(images).float(), dim=1)
        for path, feature in zip(batch["path"], image_features, strict=True):
            features[str(path)] = feature.detach().cpu()
    return features


@torch.no_grad()
def encode_denseclip_features(
    pretrained: str,
    paths: list[str],
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    feature_source: str,
) -> dict[str, torch.Tensor]:
    dataset = ImagePathDataset(paths=paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    encoder = DenseClipVisualEncoder(
        pretrained=pretrained,
        input_resolution=image_size,
        freeze=True,
    ).to(device)
    encoder.eval()
    features: dict[str, torch.Tensor] = {}
    for batch in loader:
        images = batch["image"].to(device)
        outputs = encoder(images)
        if feature_source == FEATURE_SOURCE_DENSECLIP_GLOBAL:
            vectors = outputs["global"].float()
        elif feature_source == FEATURE_SOURCE_LAYER3_GAP:
            vectors = F.adaptive_avg_pool2d(outputs["layer3"].float(), output_size=(1, 1)).flatten(1)
        else:
            raise ValueError(f"Unsupported denseclip feature source: {feature_source}")
        vectors = F.normalize(vectors, dim=1)
        for path, feature in zip(batch["path"], vectors, strict=True):
            features[str(path)] = feature.detach().cpu()
    return features


@torch.no_grad()
def encode_prompt_bank(
    model: torch.jit.ScriptModule,
    prompt_bank: dict[str, dict[str, list[str]]],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    encoded: dict[str, dict[str, object]] = {}
    for category, groups in prompt_bank.items():
        category_outputs: dict[str, object] = {}
        for group_name, prompts in groups.items():
            tokens = tokenize(prompts).to(device)
            embeddings = F.normalize(model.encode_text(tokens).float(), dim=1).detach().cpu()
            category_outputs[f"{group_name}_prompts"] = prompts
            category_outputs[f"{group_name}_embeddings"] = embeddings
        encoded[category] = category_outputs
    return encoded


def average_precision_score(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores_array)
    sorted_labels = labels_array[order]
    precision = np.cumsum(sorted_labels) / (np.arange(sorted_labels.size) + 1.0)
    return float(np.sum(precision * sorted_labels) / positives)


def choose_threshold(labels: list[int], scores: list[float]) -> dict[str, float]:
    values = sorted({float(score) for score in scores})
    if not values:
        return {"threshold": 0.0, "balanced_accuracy": 0.0, "f1": 0.0}
    candidates = [values[0] - 1e-6]
    candidates.extend(values)
    best = {
        "threshold": candidates[0],
        "balanced_accuracy": -1.0,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "specificity": 0.0,
    }
    for threshold in candidates:
        metrics = classification_metrics(labels=labels, scores=scores, threshold=float(threshold))
        if (
            metrics["balanced_accuracy"] > best["balanced_accuracy"] + 1e-12
            or (
                abs(metrics["balanced_accuracy"] - best["balanced_accuracy"]) <= 1e-12
                and metrics["f1"] > best["f1"] + 1e-12
            )
            or (
                abs(metrics["balanced_accuracy"] - best["balanced_accuracy"]) <= 1e-12
                and abs(metrics["f1"] - best["f1"]) <= 1e-12
                and metrics["accuracy"] > best["accuracy"] + 1e-12
            )
        ):
            best = {"threshold": float(threshold), **metrics}
    return best


def classification_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    preds = (scores_array >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels_array == 1)))
    tn = int(np.sum((preds == 0) & (labels_array == 0)))
    fp = int(np.sum((preds == 1) & (labels_array == 0)))
    fn = int(np.sum((preds == 0) & (labels_array == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(1, labels_array.size)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def choose_thresholds_by_category(
    records: list[dict[str, object]],
    scores: list[float],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "scores": []})
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        grouped[category]["labels"].append(int(record["label"]))
        grouped[category]["scores"].append(float(score))
    threshold_map: dict[str, dict[str, float]] = {}
    for category, payload in sorted(grouped.items()):
        threshold_map[category] = choose_threshold(payload["labels"], payload["scores"])
    return threshold_map


def prediction_array(
    records: list[dict[str, object]],
    scores: list[float],
    threshold_map: dict[str, dict[str, float]],
) -> np.ndarray:
    preds: list[int] = []
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        threshold = float(threshold_map[category]["threshold"])
        preds.append(int(float(score) >= threshold))
    return np.asarray(preds, dtype=np.int64)


def classification_metrics_from_preds(labels: list[int], preds: np.ndarray) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    tp = int(np.sum((preds == 1) & (labels_array == 1)))
    tn = int(np.sum((preds == 0) & (labels_array == 0)))
    fp = int(np.sum((preds == 1) & (labels_array == 0)))
    fn = int(np.sum((preds == 0) & (labels_array == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(1, labels_array.size)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def compute_score_batch(
    image_features: torch.Tensor,
    categories: list[str],
    prompt_embeddings: dict[str, dict[str, object]],
    p_vector: torch.Tensor | None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    scores: list[torch.Tensor] = []
    details: list[dict[str, object]] = []
    for image_feature, category in zip(image_features, categories, strict=True):
        bank = prompt_embeddings[category]
        normal_embeddings = bank["normal_embeddings"].to(image_feature.device)
        defect_embeddings = bank["defect_embeddings"].to(image_feature.device)
        if p_vector is not None:
            defect_embeddings = F.normalize(defect_embeddings + p_vector.unsqueeze(0), dim=1)
        normal_sims = image_feature @ normal_embeddings.T
        defect_sims = image_feature @ defect_embeddings.T
        max_normal_value, max_normal_idx = torch.max(normal_sims, dim=0)
        max_defect_value, max_defect_idx = torch.max(defect_sims, dim=0)
        scores.append(max_defect_value - max_normal_value)
        details.append(
            {
                "normal_score": float(max_normal_value.detach().cpu().item()),
                "max_defect_score": float(max_defect_value.detach().cpu().item()),
                "top1_prompt": bank["defect_prompts"][int(max_defect_idx.detach().cpu().item())],
                "top1_prompt_score": float(max_defect_value.detach().cpu().item()),
                "top1_normal_prompt": bank["normal_prompts"][int(max_normal_idx.detach().cpu().item())],
            }
        )
    return torch.stack(scores, dim=0), details


def evaluate_rows(
    records: list[dict[str, object]],
    scores: list[float],
    threshold_map: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        grouped[str(record["category"])].append((int(record["label"]), float(score)))

    per_category_rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        labels = [label for label, _ in items]
        category_scores = [score for _, score in items]
        threshold = float(threshold_map[category]["threshold"])
        cls_metrics = classification_metrics(labels=labels, scores=category_scores, threshold=threshold)
        per_category_rows.append(
            {
                "category": category,
                "num_query_total": len(items),
                "num_query_normal": int(sum(label == 0 for label in labels)),
                "num_query_defect": int(sum(label == 1 for label in labels)),
                "image_auroc": binary_auroc(labels, category_scores),
                "image_ap": average_precision_score(labels, category_scores),
                "accuracy": cls_metrics["accuracy"],
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "specificity": cls_metrics["specificity"],
                "balanced_accuracy": cls_metrics["balanced_accuracy"],
                "f1": cls_metrics["f1"],
                "num_pred_positive": int(cls_metrics["tp"] + cls_metrics["fp"]),
                "num_pred_negative": int(cls_metrics["tn"] + cls_metrics["fn"]),
                "threshold": threshold,
            }
        )

    labels = [int(record["label"]) for record in records]
    global_preds = prediction_array(records=records, scores=scores, threshold_map=threshold_map)
    global_cls = classification_metrics_from_preds(labels=labels, preds=global_preds)
    weak_rows = [row for row in per_category_rows if row["category"] != CONTROL_CATEGORY]
    bottle_row = next(row for row in per_category_rows if row["category"] == CONTROL_CATEGORY)
    aggregate = {
        "num_test_images": len(records),
        "num_query_normal": int(sum(label == 0 for label in labels)),
        "num_query_defect": int(sum(label == 1 for label in labels)),
        "image_auroc_mean": float(np.mean([row["image_auroc"] for row in per_category_rows])),
        "image_ap_mean": float(np.mean([row["image_ap"] for row in per_category_rows])),
        "accuracy": global_cls["accuracy"],
        "precision": global_cls["precision"],
        "recall": global_cls["recall"],
        "specificity": global_cls["specificity"],
        "balanced_accuracy": global_cls["balanced_accuracy"],
        "f1": global_cls["f1"],
        "num_pred_positive": int(global_cls["tp"] + global_cls["fp"]),
        "num_pred_negative": int(global_cls["tn"] + global_cls["fn"]),
        "weak5_image_auroc_mean": float(np.mean([row["image_auroc"] for row in weak_rows])),
        "weak5_image_ap_mean": float(np.mean([row["image_ap"] for row in weak_rows])),
        "bottle_image_auroc": float(bottle_row["image_auroc"]),
    }
    return per_category_rows, aggregate


def write_summary(path: Path, experiments_rows: list[dict[str, object]]) -> None:
    lines = ["# Prompt Text Summary", ""]
    for row in experiments_rows:
        lines.append(
            f"{row['experiment']}: auroc={row['image_auroc_mean']:.6f} ap={row['image_ap_mean']:.6f} "
            f"acc={row['accuracy']:.6f} bal_acc={row['balanced_accuracy']:.6f} f1={row['f1']:.6f}"
        )
    if len(experiments_rows) >= 2:
        base = experiments_rows[0]
        for row in experiments_rows[1:]:
            lines.append(
                f"delta_vs_E0 {row['experiment']}: "
                f"auroc={row['image_auroc_mean'] - base['image_auroc_mean']:+.6f} "
                f"ap={row['image_ap_mean'] - base['image_ap_mean']:+.6f} "
                f"weak5_auroc={row['weak5_image_auroc_mean'] - base['weak5_image_auroc_mean']:+.6f} "
                f"bottle_auroc={row['bottle_image_auroc'] - base['bottle_image_auroc']:+.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_feature_matrix(
    records: list[dict[str, object]],
    features_by_path: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.stack([features_by_path[str(record["path"])] for record in records], dim=0)


def mean_score_lists(score_lists: list[list[float]]) -> list[float]:
    if not score_lists:
        return []
    stacked = np.asarray(score_lists, dtype=np.float64)
    return stacked.mean(axis=0).astype(np.float64).tolist()


def build_predictions_rows(
    experiment: str,
    records: list[dict[str, object]],
    scores: list[float],
    details: list[dict[str, object]],
    threshold_map: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record, score, detail in zip(records, scores, details, strict=True):
        category = str(record["category"])
        threshold = float(threshold_map[category]["threshold"])
        rows.append(
            {
                "experiment": experiment,
                "path": str(record["path"]),
                "category": category,
                "defect_type": str(record.get("defect_type") or ""),
                "label": int(record["label"]),
                "score": float(score),
                "pred_label": int(float(score) >= threshold),
                "threshold": float(threshold),
                "normal_score": detail.get("normal_score", ""),
                "max_defect_score": detail.get("max_defect_score", ""),
                "top1_prompt": detail.get("top1_prompt", ""),
                "top1_prompt_score": detail.get("top1_prompt_score", ""),
            }
        )
    return rows


def append_result_rows(
    experiments_rows: list[dict[str, object]],
    per_category_rows: list[dict[str, object]],
    predictions_rows: list[dict[str, object]],
    experiment_name: str,
    seed: int,
    aggregate: dict[str, float],
    per_category: list[dict[str, object]],
    predictions: list[dict[str, object]],
    selection_source: str,
    threshold_source: str,
    threshold_value: object,
    scope: str,
    selection_epoch: str = "",
) -> None:
    row = {
        "experiment": experiment_name,
        "track": "prompt-text",
        "scope": scope,
        "seed": seed,
        "num_categories": len(per_category),
        **aggregate,
        "selection_source": selection_source,
        "threshold_source": threshold_source,
        "threshold_value": threshold_value,
    }
    if selection_epoch:
        row["selection_epoch"] = selection_epoch
    experiments_rows.append(row)
    for item in per_category:
        per_category_rows.append({"experiment": experiment_name, **item})
    predictions_rows.extend(predictions)


def train_split_model(
    split_index: int,
    source_name: str,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_categories: list[str],
    holdout_features: torch.Tensor,
    holdout_labels: list[int],
    holdout_categories: list[str],
    holdout_records: list[dict[str, object]],
    prompt_embeddings: dict[str, dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict[str, object]:
    train_features = train_features.float()
    train_labels = train_labels.float()
    holdout_features = holdout_features.float()

    p_vector = torch.nn.Parameter(torch.zeros(train_features.shape[1], dtype=torch.float32, device=device))
    optimizer = AdamW([p_vector], lr=args.lr, weight_decay=args.weight_decay)
    category_names = sorted(prompt_embeddings.keys())
    category_to_index = {name: idx for idx, name in enumerate(category_names)}
    train_category_indices = torch.tensor([category_to_index[name] for name in train_categories], dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_features, train_labels, train_category_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )

    holdout_features_device = holdout_features.to(device)
    holdout_best_metric = float("-inf")
    best_state = p_vector.detach().cpu().clone()
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        steps = 0
        for batch_features, batch_labels, batch_category_indices in loader:
            batch_categories = [category_names[int(idx)] for idx in batch_category_indices.tolist()]
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_scores, _ = compute_score_batch(
                image_features=batch_features,
                categories=batch_categories,
                prompt_embeddings=prompt_embeddings,
                p_vector=p_vector,
            )
            bce = F.binary_cross_entropy_with_logits(batch_scores, batch_labels)
            l2 = args.p_l2 * torch.sum(p_vector * p_vector)
            loss = bce + l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        holdout_scores_tensor, _ = compute_score_batch(
            image_features=holdout_features_device,
            categories=holdout_categories,
            prompt_embeddings=prompt_embeddings,
            p_vector=p_vector,
        )
        holdout_scores = holdout_scores_tensor.detach().cpu().tolist()
        holdout_threshold_map = choose_thresholds_by_category(records=holdout_records, scores=holdout_scores)
        _, holdout_aggregate = evaluate_rows(
            records=holdout_records,
            scores=holdout_scores,
            threshold_map=holdout_threshold_map,
        )
        holdout_metric = float(holdout_aggregate["image_auroc_mean"])
        metric_row = {
            "source": source_name,
            "split_index": split_index,
            "epoch": epoch,
            "loss": running_loss / max(1, steps),
            "holdout_image_auroc_mean": holdout_metric,
            "holdout_image_ap_mean": float(holdout_aggregate["image_ap_mean"]),
            "p_norm": float(torch.norm(p_vector.detach()).cpu().item()),
        }
        append_jsonl(metrics_path, metric_row)
        if holdout_metric > holdout_best_metric + args.min_delta:
            holdout_best_metric = holdout_metric
            best_state = p_vector.detach().cpu().clone()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_holdout_metric": holdout_best_metric,
    }


def compute_scores_for_records(
    features_by_path: dict[str, torch.Tensor],
    records: list[dict[str, object]],
    prompt_embeddings: dict[str, dict[str, object]],
    p_vector: torch.Tensor | None,
    device: torch.device,
) -> tuple[list[float], list[dict[str, object]]]:
    feature_tensor = build_feature_matrix(records, features_by_path).to(device)
    categories = [str(record["category"]) for record in records]
    score_tensor, details = compute_score_batch(
        image_features=feature_tensor,
        categories=categories,
        prompt_embeddings=prompt_embeddings,
        p_vector=p_vector,
    )
    return score_tensor.detach().cpu().tolist(), details


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scope = args.scope.strip() or cache_dir.parent.name or cache_dir.name

    feature_sources = parse_feature_sources(args.feature_sources)
    records, _ = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    resplits = build_resplits(
        train_records=train_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        num_resplits=args.num_resplits,
    )

    clip_model = torch.jit.load(args.pretrained, map_location=device).eval()
    if device.type == "cuda":
        clip_model = clip_model.to(device)

    all_paths = sorted({str(record["path"]) for record in records})
    feature_maps: dict[str, dict[str, torch.Tensor]] = {}
    if FEATURE_SOURCE_CLIP_GLOBAL in feature_sources:
        feature_maps[FEATURE_SOURCE_CLIP_GLOBAL] = encode_clip_global_features(
            model=clip_model,
            paths=all_paths,
            image_size=args.clip_image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
        )
    denseclip_sources = [source for source in feature_sources if source != FEATURE_SOURCE_CLIP_GLOBAL]
    for source in denseclip_sources:
        feature_maps[source] = encode_denseclip_features(
            pretrained=args.pretrained,
            paths=all_paths,
            image_size=args.denseclip_image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            feature_source=source,
        )

    prompt_bank = build_prompt_bank(records)
    prompt_embeddings = encode_prompt_bank(model=clip_model, prompt_bank=prompt_bank, device=device)

    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    config = {
        "track": "new-branch/prompt-text",
        "scope": scope,
        "cache_dir": str(cache_dir),
        "pretrained": args.pretrained,
        "output_dir": str(output_dir),
        "clip_image_size": args.clip_image_size,
        "denseclip_image_size": args.denseclip_image_size,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "num_resplits": args.num_resplits,
        "feature_sources": feature_sources,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "p_l2": args.p_l2,
        "score_contract": "max_defect_similarity - max_normal_similarity",
        "visual_side": "frozen",
        "selection_contract": "per_split_best_epoch + per_category_pooled_holdout_threshold + averaged_query_logits",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    split_summary = {
        "num_query_eval": len(eval_records),
        "num_resplits": args.num_resplits,
        "resplits": [],
    }
    for resplit in resplits:
        split_entry = {
            "split_index": resplit["split_index"],
            "split_seed": resplit["split_seed"],
            "num_support_train": len(resplit["support_train"]),
            "num_support_holdout": len(resplit["holdout"]),
            "support_train_by_category_label": {},
            "support_holdout_by_category_label": {},
        }
        for key_name, subset_name in (
            ("support_train_by_category_label", "support_train"),
            ("support_holdout_by_category_label", "holdout"),
        ):
            counts: dict[str, int] = defaultdict(int)
            for record in resplit[subset_name]:
                counts[f"{record['category']}|label={record['label']}"] += 1
            split_entry[key_name] = dict(sorted(counts.items()))
        split_summary["resplits"].append(split_entry)
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    prompt_stats = {}
    for category, bank in prompt_embeddings.items():
        prompt_stats[category] = {
            "num_normal_prompts": len(bank["normal_prompts"]),
            "num_defect_prompts": len(bank["defect_prompts"]),
            "normal_prompts": bank["normal_prompts"],
            "defect_prompts": bank["defect_prompts"],
        }
    (output_dir / "prompt_stats.json").write_text(json.dumps(prompt_stats, indent=2), encoding="utf-8")

    experiments_rows: list[dict[str, object]] = []
    per_category_rows: list[dict[str, object]] = []
    predictions_rows: list[dict[str, object]] = []

    pooled_holdout_baseline_scores: list[float] = []
    pooled_holdout_baseline_labels: list[int] = []
    for resplit in resplits:
        holdout = resplit["holdout"]
        pooled_holdout_baseline_scores.extend(float(record["winner_image_score"]) for record in holdout)
        pooled_holdout_baseline_labels.extend(int(record["label"]) for record in holdout)
    baseline_scores = [float(record["winner_image_score"]) for record in eval_records]
    baseline_threshold_map = choose_thresholds_by_category(records=sum([resplit["holdout"] for resplit in resplits], []), scores=pooled_holdout_baseline_scores)
    e0_per_category, e0_aggregate = evaluate_rows(
        records=eval_records,
        scores=baseline_scores,
        threshold_map=baseline_threshold_map,
    )
    e0_predictions = build_predictions_rows(
        experiment="E0",
        records=eval_records,
        scores=baseline_scores,
        details=[{} for _ in eval_records],
        threshold_map=baseline_threshold_map,
    )
    append_result_rows(
        experiments_rows=experiments_rows,
        per_category_rows=per_category_rows,
        predictions_rows=predictions_rows,
        experiment_name="E0",
        seed=args.seed,
        aggregate=e0_aggregate,
        per_category=e0_per_category,
        predictions=e0_predictions,
        selection_source="frozen_stage2_winner",
        threshold_source="per_category_pooled_multisplit_holdout_best_balanced_accuracy",
        threshold_value="per_category",
        scope=scope,
    )

    resplit_history: dict[str, object] = {
        "threshold_pooling": "per_category_pooled_multisplit_holdout_best_balanced_accuracy",
        "feature_sources": {},
    }
    threshold_report: dict[str, object] = {
        "E0": baseline_threshold_map,
    }

    for source_name in feature_sources:
        features_by_path = feature_maps[source_name]
        feature_history: dict[str, object] = {"frozen": {"splits": []}, "prompt_plus_p": {"splits": []}}

        pooled_frozen_holdout_scores: list[float] = []
        pooled_frozen_holdout_labels: list[int] = []
        frozen_eval_scores, frozen_eval_details = compute_scores_for_records(
            features_by_path=features_by_path,
            records=eval_records,
            prompt_embeddings=prompt_embeddings,
            p_vector=None,
            device=device,
        )
        for resplit in resplits:
            holdout_scores, _ = compute_scores_for_records(
                features_by_path=features_by_path,
                records=resplit["holdout"],
                prompt_embeddings=prompt_embeddings,
                p_vector=None,
                device=device,
            )
            holdout_labels = [int(record["label"]) for record in resplit["holdout"]]
            pooled_frozen_holdout_scores.extend(holdout_scores)
            pooled_frozen_holdout_labels.extend(holdout_labels)
            feature_history["frozen"]["splits"].append(
                {
                    "split_index": resplit["split_index"],
                    "split_seed": resplit["split_seed"],
                    "num_holdout": len(resplit["holdout"]),
                    "holdout_image_auroc_mean": binary_auroc(holdout_labels, holdout_scores),
                }
            )
        frozen_threshold_map = choose_thresholds_by_category(
            records=sum([resplit["holdout"] for resplit in resplits], []),
            scores=pooled_frozen_holdout_scores,
        )
        frozen_per_category, frozen_aggregate = evaluate_rows(
            records=eval_records,
            scores=frozen_eval_scores,
            threshold_map=frozen_threshold_map,
        )
        frozen_predictions = build_predictions_rows(
            experiment=f"prompt_only_frozen__{source_name}",
            records=eval_records,
            scores=frozen_eval_scores,
            details=frozen_eval_details,
            threshold_map=frozen_threshold_map,
        )
        append_result_rows(
            experiments_rows=experiments_rows,
            per_category_rows=per_category_rows,
            predictions_rows=predictions_rows,
            experiment_name=f"prompt_only_frozen__{source_name}",
            seed=args.seed,
            aggregate=frozen_aggregate,
            per_category=frozen_per_category,
            predictions=frozen_predictions,
            selection_source="frozen_prompt_bank",
            threshold_source="per_category_pooled_multisplit_holdout_best_balanced_accuracy",
            threshold_value="per_category",
            scope=scope,
        )
        threshold_report[f"prompt_only_frozen__{source_name}"] = frozen_threshold_map

        split_eval_scores: list[list[float]] = []
        split_eval_details: list[list[dict[str, object]]] = []
        pooled_plus_holdout_scores: list[float] = []
        pooled_plus_holdout_labels: list[int] = []
        selection_epochs: list[str] = []
        for resplit in resplits:
            support_train = resplit["support_train"]
            holdout = resplit["holdout"]
            train_features = build_feature_matrix(support_train, features_by_path)
            train_labels = torch.tensor([float(record["label"]) for record in support_train], dtype=torch.float32)
            train_categories = [str(record["category"]) for record in support_train]
            holdout_features = build_feature_matrix(holdout, features_by_path)
            holdout_labels = [int(record["label"]) for record in holdout]
            holdout_categories = [str(record["category"]) for record in holdout]

            split_result = train_split_model(
                split_index=int(resplit["split_index"]),
                source_name=source_name,
                train_features=train_features,
                train_labels=train_labels,
                train_categories=train_categories,
                holdout_features=holdout_features,
                holdout_labels=holdout_labels,
                holdout_categories=holdout_categories,
                holdout_records=holdout,
                prompt_embeddings=prompt_embeddings,
                device=device,
                args=args,
                metrics_path=metrics_path,
            )
            best_state = split_result["best_state"].to(device)
            best_epoch = int(split_result["best_epoch"])
            selection_epochs.append(str(best_epoch))

            holdout_scores, _ = compute_scores_for_records(
                features_by_path=features_by_path,
                records=holdout,
                prompt_embeddings=prompt_embeddings,
                p_vector=best_state,
                device=device,
            )
            pooled_plus_holdout_scores.extend(holdout_scores)
            pooled_plus_holdout_labels.extend(holdout_labels)

            eval_scores, eval_details = compute_scores_for_records(
                features_by_path=features_by_path,
                records=eval_records,
                prompt_embeddings=prompt_embeddings,
                p_vector=best_state,
                device=device,
            )
            split_eval_scores.append(eval_scores)
            split_eval_details.append(eval_details)
            feature_history["prompt_plus_p"]["splits"].append(
                {
                    "split_index": resplit["split_index"],
                    "split_seed": resplit["split_seed"],
                    "best_epoch": best_epoch,
                    "best_holdout_metric": float(split_result["best_holdout_metric"]),
                    "num_support_train": len(support_train),
                    "num_holdout": len(holdout),
                }
            )

        plus_threshold_map = choose_thresholds_by_category(
            records=sum([resplit["holdout"] for resplit in resplits], []),
            scores=pooled_plus_holdout_scores,
        )
        plus_eval_scores = mean_score_lists(split_eval_scores)
        representative_details = split_eval_details[0] if split_eval_details else [{} for _ in eval_records]
        plus_per_category, plus_aggregate = evaluate_rows(
            records=eval_records,
            scores=plus_eval_scores,
            threshold_map=plus_threshold_map,
        )
        plus_predictions = build_predictions_rows(
            experiment=f"prompt_plus_p__{source_name}",
            records=eval_records,
            scores=plus_eval_scores,
            details=representative_details,
            threshold_map=plus_threshold_map,
        )
        append_result_rows(
            experiments_rows=experiments_rows,
            per_category_rows=per_category_rows,
            predictions_rows=predictions_rows,
            experiment_name=f"prompt_plus_p__{source_name}",
            seed=args.seed,
            aggregate=plus_aggregate,
            per_category=plus_per_category,
            predictions=plus_predictions,
            selection_source="multisplit_holdout_image_auroc_mean",
            threshold_source="per_category_pooled_multisplit_holdout_best_balanced_accuracy",
            threshold_value="per_category",
            scope=scope,
            selection_epoch=",".join(selection_epochs),
        )
        resplit_history["feature_sources"][source_name] = feature_history
        threshold_report[f"prompt_plus_p__{source_name}"] = plus_threshold_map

    (output_dir / "resplit_history.json").write_text(json.dumps(resplit_history, indent=2), encoding="utf-8")
    (output_dir / "threshold_report.json").write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "predictions.csv", predictions_rows)
    write_summary(output_dir / "summary.md", experiments_rows)


if __name__ == "__main__":
    main()
