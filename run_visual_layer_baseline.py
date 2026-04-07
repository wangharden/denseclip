import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.data import ImageTransform
from fewshot.feature_bank import build_reference_bank, flatten_feature_map
from fewshot.scoring import (
    AGGREGATION_MODE_TOPK_MEAN,
    SCORE_MODE_DEFECT_MINUS_NORMAL,
    aggregate_image_score,
    build_score_map,
    compute_similarity_maps,
)
from fewshot.stage_a1 import binary_auroc
from run_stage3_head import load_records, resolve_device, set_seed, split_records, write_csv


CONTROL_CATEGORY = "bottle"
EXPERIMENT_LAYER4_GLOBAL = "layer4_global"
EXPERIMENT_LAYER3_GAP = "layer3_gap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen layer3/layer4 image-level baselines under a shared split.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/new-branch/layer-baseline/weak5_bottle/seed42/layer3_only_v1",
    )
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--reference-topk", type=int, default=3)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.transform = ImageTransform(image_size=image_size, augment=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        return {"path": path, "image": self.transform(Path(path))}


def split_support_holdout(
    train_records: list[dict[str, object]],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.RandomState(seed)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in train_records:
        key = (str(record["category"]), int(record["label"]))
        grouped.setdefault(key, []).append(record)

    support_train: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: str(item["path"]))
        holdout_count = int(round(len(group) * holdout_fraction))
        holdout_count = max(1, holdout_count)
        holdout_count = min(len(group) - 1, holdout_count)
        holdout_ids = set(rng.permutation(len(group))[:holdout_count].tolist())
        for index, record in enumerate(group):
            if index in holdout_ids:
                holdout.append(record)
            else:
                support_train.append(record)

    if not support_train or not holdout:
        raise ValueError("Support holdout split produced an empty support-train or holdout partition.")
    return support_train, holdout


@torch.no_grad()
def encode_features(
    encoder: DenseClipVisualEncoder,
    paths: list[str],
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    dataset = ImagePathDataset(paths=paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for batch in loader:
        images = batch["image"].to(device)
        encoded = encoder(images)
        layer3 = F.normalize(encoded["layer3"].float(), dim=1).cpu()
        global_feat = F.normalize(encoded["global"].float(), dim=1).cpu()
        for path, layer3_item, global_item in zip(batch["path"], layer3, global_feat, strict=True):
            outputs[str(path)] = {
                "layer3": layer3_item.contiguous(),
                "global": global_item.contiguous(),
            }
    return outputs


def build_reference_payloads(
    support_train: list[dict[str, object]],
    encoded_by_path: dict[str, dict[str, torch.Tensor]],
    reference_topk: int,
) -> dict[str, dict[str, torch.Tensor | int]]:
    grouped: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    for record in support_train:
        path = str(record["path"])
        category = str(record["category"])
        key = "normal" if int(record["label"]) == 0 else "defect"
        grouped[category][f"{key}_layer3"].append(encoded_by_path[path]["layer3"])
        grouped[category][f"{key}_global"].append(encoded_by_path[path]["global"])

    payloads: dict[str, dict[str, torch.Tensor | int]] = {}
    for category, group in grouped.items():
        normal_layer3 = torch.stack(group["normal_layer3"], dim=0)
        defect_layer3 = torch.stack(group["defect_layer3"], dim=0)
        normal_global = torch.stack(group["normal_global"], dim=0)
        defect_global = torch.stack(group["defect_global"], dim=0)
        payloads[category] = {
            "layer3_normal_ref": build_reference_bank(
                features=flatten_feature_map(normal_layer3),
                prototype_family="memory_bank",
                num_prototypes=1,
                seed=42,
                num_iters=1,
            ).contiguous(),
            "layer3_defect_ref": build_reference_bank(
                features=flatten_feature_map(defect_layer3),
                prototype_family="memory_bank",
                num_prototypes=1,
                seed=43,
                num_iters=1,
            ).contiguous(),
            "global_normal_ref": F.normalize(normal_global.mean(dim=0, keepdim=True), dim=1)[0].contiguous(),
            "global_defect_ref": F.normalize(defect_global.mean(dim=0, keepdim=True), dim=1)[0].contiguous(),
            "reference_topk": int(reference_topk),
            "num_support_normal": int(normal_global.shape[0]),
            "num_support_defect": int(defect_global.shape[0]),
        }
    return payloads


def score_layer4_global(
    record: dict[str, object],
    encoded_by_path: dict[str, dict[str, torch.Tensor]],
    refs: dict[str, dict[str, torch.Tensor | int]],
) -> float:
    category = str(record["category"])
    path = str(record["path"])
    query = encoded_by_path[path]["global"]
    normal_ref = refs[category]["global_normal_ref"]
    defect_ref = refs[category]["global_defect_ref"]
    normal_sim = float(torch.dot(query, normal_ref).item())  # type: ignore[arg-type]
    defect_sim = float(torch.dot(query, defect_ref).item())  # type: ignore[arg-type]
    return defect_sim - normal_sim


def score_layer3_gap(
    record: dict[str, object],
    encoded_by_path: dict[str, dict[str, torch.Tensor]],
    refs: dict[str, dict[str, torch.Tensor | int]],
    topk_ratio: float,
) -> float:
    category = str(record["category"])
    path = str(record["path"])
    layer3 = encoded_by_path[path]["layer3"].unsqueeze(0)
    similarity_maps = compute_similarity_maps(
        feature_map=layer3,
        normal_prototype=refs[category]["layer3_normal_ref"],  # type: ignore[arg-type]
        defect_prototype=refs[category]["layer3_defect_ref"],  # type: ignore[arg-type]
        reference_topk=int(refs[category]["reference_topk"]),
    )
    score_map = build_score_map(similarity_maps, score_mode=SCORE_MODE_DEFECT_MINUS_NORMAL)
    return float(
        aggregate_image_score(
            score_map,
            aggregation_mode=AGGREGATION_MODE_TOPK_MEAN,
            topk_ratio=topk_ratio,
        )[0].item()
    )


def average_precision(labels: list[int], scores: list[float]) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    if positives <= 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y_true[order]
    tp = np.cumsum(ranked == 1)
    fp = np.cumsum(ranked == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[ranked == 1].sum() / positives)


def classification_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def select_threshold(labels: list[int], scores: list[float]) -> dict[str, float]:
    unique_scores = sorted({float(score) for score in scores})
    if not unique_scores:
        return {"threshold": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    candidates = [unique_scores[0] - 1e-6]
    candidates.extend(unique_scores)
    best = {"threshold": candidates[0], "accuracy": -1.0, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in candidates:
        metrics = classification_metrics(labels, scores, threshold)
        if (
            metrics["f1"] > best["f1"] + 1e-12
            or (
                abs(metrics["f1"] - best["f1"]) <= 1e-12
                and metrics["accuracy"] > best["accuracy"] + 1e-12
            )
        ):
            best = {"threshold": float(threshold), **metrics}
    return best


def evaluate_per_category(
    records: list[dict[str, object]],
    scores: list[float],
    threshold: float,
    experiment: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        grouped[str(record["category"])].append((int(record["label"]), float(score)))
    rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        labels = [label for label, _ in items]
        category_scores = [score for _, score in items]
        cls_metrics = classification_metrics(labels, category_scores, threshold)
        rows.append(
            {
                "experiment": experiment,
                "category": category,
                "num_test_images": len(items),
                "num_positive_images": int(sum(labels)),
                "num_negative_images": int(len(labels) - sum(labels)),
                "image_auroc": binary_auroc(labels, category_scores),
                "image_ap": average_precision(labels, category_scores),
                "accuracy": cls_metrics["accuracy"],
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "f1": cls_metrics["f1"],
                "threshold": float(threshold),
            }
        )
    return rows


def summarize_experiment(
    experiment: str,
    per_category_rows: list[dict[str, object]],
    labels: list[int],
    scores: list[float],
    threshold_info: dict[str, float],
) -> dict[str, object]:
    weak_rows = [row for row in per_category_rows if str(row["category"]) != CONTROL_CATEGORY]
    bottle_row = next(row for row in per_category_rows if str(row["category"]) == CONTROL_CATEGORY)
    query_metrics = classification_metrics(labels, scores, float(threshold_info["threshold"]))
    return {
        "experiment": experiment,
        "num_test_images": len(labels),
        "image_auroc_mean": float(np.mean([float(row["image_auroc"]) for row in per_category_rows])),
        "image_ap_mean": float(np.mean([float(row["image_ap"]) for row in per_category_rows])),
        "accuracy": query_metrics["accuracy"],
        "precision": query_metrics["precision"],
        "recall": query_metrics["recall"],
        "f1": query_metrics["f1"],
        "weak5_image_auroc_mean": float(np.mean([float(row["image_auroc"]) for row in weak_rows])),
        "weak5_image_ap_mean": float(np.mean([float(row["image_ap"]) for row in weak_rows])),
        "bottle_image_auroc": float(bottle_row["image_auroc"]),
        "selection_source": "frozen_fixed_control_vs_candidate",
        "threshold_source": "holdout_max_f1",
        "threshold": float(threshold_info["threshold"]),
    }


def build_predictions_rows(
    records: list[dict[str, object]],
    scores: list[float],
    threshold: float,
    experiment: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record, score in zip(records, scores, strict=True):
        rows.append(
            {
                "experiment": experiment,
                "category": str(record["category"]),
                "path": str(record["path"]),
                "label": int(record["label"]),
                "defect_type": str(record["defect_type"]),
                "image_score": float(score),
                "threshold": float(threshold),
                "prediction": int(float(score) >= threshold),
            }
        )
    return rows


def write_summary(path: Path, experiments_rows: list[dict[str, object]]) -> None:
    control = next(row for row in experiments_rows if row["experiment"] == EXPERIMENT_LAYER4_GLOBAL)
    candidate = next(row for row in experiments_rows if row["experiment"] == EXPERIMENT_LAYER3_GAP)
    lines = [
        "# Layer Baseline Summary",
        "",
        f"{EXPERIMENT_LAYER4_GLOBAL}: "
        f"mean_auroc={control['image_auroc_mean']:.4f} "
        f"mean_ap={control['image_ap_mean']:.4f} "
        f"weak5_auroc={control['weak5_image_auroc_mean']:.4f} "
        f"bottle_auroc={control['bottle_image_auroc']:.4f}",
        f"{EXPERIMENT_LAYER3_GAP}: "
        f"mean_auroc={candidate['image_auroc_mean']:.4f} "
        f"mean_ap={candidate['image_ap_mean']:.4f} "
        f"weak5_auroc={candidate['weak5_image_auroc_mean']:.4f} "
        f"bottle_auroc={candidate['bottle_image_auroc']:.4f}",
        "",
        f"delta_mean_auroc={candidate['image_auroc_mean'] - control['image_auroc_mean']:+.4f}",
        f"delta_mean_ap={candidate['image_ap_mean'] - control['image_ap_mean']:+.4f}",
        f"delta_weak5_auroc={candidate['weak5_image_auroc_mean'] - control['weak5_image_auroc_mean']:+.4f}",
        f"delta_bottle_auroc={candidate['bottle_image_auroc'] - control['bottle_image_auroc']:+.4f}",
        "",
        f"selection_source={candidate['selection_source']}",
        f"threshold_source={candidate['threshold_source']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, _ = load_records(Path(args.cache_dir))
    train_records, eval_records = split_records(records)
    support_train, holdout_records = split_support_holdout(
        train_records=train_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    all_paths = sorted(
        {
            *(str(record["path"]) for record in support_train),
            *(str(record["path"]) for record in holdout_records),
            *(str(record["path"]) for record in eval_records),
        }
    )
    encoder = DenseClipVisualEncoder(
        pretrained=args.pretrained,
        input_resolution=args.image_size,
        freeze=True,
    ).to(device)
    encoder.eval()
    encoded_by_path = encode_features(
        encoder=encoder,
        paths=all_paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    refs = build_reference_payloads(
        support_train=support_train,
        encoded_by_path=encoded_by_path,
        reference_topk=args.reference_topk,
    )

    experiments_rows: list[dict[str, object]] = []
    per_category_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    feature_stats = {
        "num_categories": len(refs),
        "categories": {
            category: {
                "num_support_normal": int(payload["num_support_normal"]),
                "num_support_defect": int(payload["num_support_defect"]),
                "layer3_reference_bank_size": int(payload["layer3_normal_ref"].shape[0]),  # type: ignore[index]
                "layer3_defect_bank_size": int(payload["layer3_defect_ref"].shape[0]),  # type: ignore[index]
                "global_dim": int(payload["global_normal_ref"].shape[0]),  # type: ignore[index]
            }
            for category, payload in refs.items()
        },
    }

    experiment_fns = {
        EXPERIMENT_LAYER4_GLOBAL: lambda record: score_layer4_global(record, encoded_by_path, refs),
        EXPERIMENT_LAYER3_GAP: lambda record: score_layer3_gap(record, encoded_by_path, refs, args.topk_ratio),
    }

    for experiment, score_fn in experiment_fns.items():
        holdout_scores = [float(score_fn(record)) for record in holdout_records]
        holdout_labels = [int(record["label"]) for record in holdout_records]
        threshold_info = select_threshold(holdout_labels, holdout_scores)
        eval_scores = [float(score_fn(record)) for record in eval_records]
        eval_labels = [int(record["label"]) for record in eval_records]
        experiment_per_category = evaluate_per_category(
            records=eval_records,
            scores=eval_scores,
            threshold=float(threshold_info["threshold"]),
            experiment=experiment,
        )
        experiments_rows.append(
            summarize_experiment(
                experiment=experiment,
                per_category_rows=experiment_per_category,
                labels=eval_labels,
                scores=eval_scores,
                threshold_info=threshold_info,
            )
        )
        per_category_rows.extend(experiment_per_category)
        prediction_rows.extend(
            build_predictions_rows(
                records=eval_records,
                scores=eval_scores,
                threshold=float(threshold_info["threshold"]),
                experiment=experiment,
            )
        )

    split_summary = {
        "selection_source": "shared_holdout_threshold_only",
        "threshold_source": "holdout_max_f1",
        "num_support_train": len(support_train),
        "num_holdout": len(holdout_records),
        "num_test_images": len(eval_records),
        "categories": {
            category: {
                "support_train": int(
                    sum(1 for record in support_train if str(record["category"]) == category)
                ),
                "holdout": int(
                    sum(1 for record in holdout_records if str(record["category"]) == category)
                ),
                "query_eval": int(
                    sum(1 for record in eval_records if str(record["category"]) == category)
                ),
            }
            for category in sorted({str(record["category"]) for record in records})
        },
    }

    config = {
        **vars(args),
        "track": "new-branch layer-baseline",
        "experiments": [EXPERIMENT_LAYER4_GLOBAL, EXPERIMENT_LAYER3_GAP],
        "control_experiment": EXPERIMENT_LAYER4_GLOBAL,
        "candidate_experiment": EXPERIMENT_LAYER3_GAP,
        "selection_source": "frozen_fixed_control_vs_candidate",
        "threshold_source": "holdout_max_f1",
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    (output_dir / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_summary(output_dir / "summary.md", experiments_rows)
    print(json.dumps({"output_dir": str(output_dir), "num_experiments": len(experiments_rows)}, indent=2))


if __name__ == "__main__":
    main()
