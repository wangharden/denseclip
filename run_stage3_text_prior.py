import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_ROOT = REPO_ROOT / "detection" / "denseclip"
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

if "ftfy" not in sys.modules:
    ftfy_stub = types.ModuleType("ftfy")
    ftfy_stub.fix_text = lambda text: text
    sys.modules["ftfy"] = ftfy_stub

from untils import tokenize

from fewshot.data import ImageTransform
from fewshot.stage_a1 import binary_auroc
from run_stage3_head import load_records, resolve_device, split_records, write_csv


CONTROL_CATEGORY = "bottle"
DEFAULT_NORMAL_TEMPLATES = (
    "a photo of a normal {category}.",
    "a photo of a good {category}.",
    "a photo of a flawless {category}.",
)
DEFAULT_ANOMALY_TEMPLATES = (
    "a photo of an anomalous {category}.",
    "a photo of a defective {category}.",
    "a photo of a damaged {category}.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 frozen text-prior image-level smoke on cached Stage2 winner outputs.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--output-dir", default="outputs/stage3/p3_text_prior/weak5_bottle/seed42_monotonic")
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--beta-max", type=float, default=2.0)
    parser.add_argument("--beta-steps", type=int, default=21)
    parser.add_argument("--tau-quantiles", default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--bottle-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--weak5-min-delta", type=float, default=0.01)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    train_split: list[dict[str, object]] = []
    holdout_split: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: str(item["path"]))
        holdout_count = int(round(len(group) * holdout_fraction))
        holdout_count = max(1, holdout_count)
        holdout_count = min(len(group) - 1, holdout_count)
        holdout_ids = set(rng.permutation(len(group))[:holdout_count].tolist())
        for index, record in enumerate(group):
            if index in holdout_ids:
                holdout_split.append(record)
            else:
                train_split.append(record)
    if not train_split or not holdout_split:
        raise ValueError("Support holdout split produced an empty train or holdout partition.")
    return train_split, holdout_split


def evaluate_image_per_category(
    records: list[dict[str, object]],
    image_scores: list[float],
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for record, score in zip(records, image_scores, strict=True):
        grouped.setdefault(str(record["category"]), []).append((int(record["label"]), float(score)))

    rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        labels = [label for label, _ in items]
        scores = [score for _, score in items]
        rows.append({"category": category, "image_auroc": binary_auroc(labels, scores)})
    return rows


def summarize_image_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    weak_rows = [row for row in rows if str(row["category"]) != CONTROL_CATEGORY]
    bottle_row = next(row for row in rows if str(row["category"]) == CONTROL_CATEGORY)
    return {
        "overall_image_auroc_mean": float(np.mean([row["image_auroc"] for row in rows])),
        "weak5_image_auroc_mean": float(np.mean([row["image_auroc"] for row in weak_rows])),
        "bottle_image_auroc": float(bottle_row["image_auroc"]),
    }


def build_text_prompt_map(categories: set[str]) -> dict[str, dict[str, list[str]]]:
    prompt_map: dict[str, dict[str, list[str]]] = {}
    for category in sorted(categories):
        prompt_map[category] = {
            "normal": [template.format(category=category) for template in DEFAULT_NORMAL_TEMPLATES],
            "anomaly": [template.format(category=category) for template in DEFAULT_ANOMALY_TEMPLATES],
        }
    return prompt_map


@torch.no_grad()
def encode_image_features(
    model: torch.jit.ScriptModule,
    paths: list[str],
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = ImagePathDataset(paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    features: dict[str, torch.Tensor] = {}
    for batch in loader:
        images = batch["image"].to(device)
        image_features = F.normalize(model.encode_image(images).float(), dim=1)
        for path, feature in zip(batch["path"], image_features, strict=True):
            features[str(path)] = feature.detach().cpu()
    return features


@torch.no_grad()
def encode_text_prototypes(
    model: torch.jit.ScriptModule,
    prompt_map: dict[str, dict[str, list[str]]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    prototypes: dict[str, dict[str, torch.Tensor]] = {}
    for category, prompts in prompt_map.items():
        category_outputs: dict[str, torch.Tensor] = {}
        for group_name, prompt_list in prompts.items():
            tokens = tokenize(prompt_list).to(device)
            text_features = F.normalize(model.encode_text(tokens).float(), dim=1)
            prototype = F.normalize(text_features.mean(dim=0, keepdim=True), dim=1)[0].detach().cpu()
            category_outputs[group_name] = prototype
        prototypes[category] = category_outputs
    return prototypes


def compute_text_gaps(
    records: list[dict[str, object]],
    image_features: dict[str, torch.Tensor],
    text_prototypes: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    text_gap_by_path: dict[str, float] = {}
    for record in records:
        path = str(record["path"])
        category = str(record["category"])
        image_feature = image_features[path]
        normal_feature = text_prototypes[category]["normal"]
        anomaly_feature = text_prototypes[category]["anomaly"]
        normal_sim = float(torch.dot(image_feature, normal_feature).item())
        anomaly_sim = float(torch.dot(image_feature, anomaly_feature).item())
        text_gap_by_path[path] = anomaly_sim - normal_sim
    return text_gap_by_path


def apply_monotonic_update(
    base_scores: list[float],
    text_gaps: list[float],
    beta: float,
    tau: float,
) -> list[float]:
    base_array = np.asarray(base_scores, dtype=np.float64)
    gap_array = np.asarray(text_gaps, dtype=np.float64)
    update = np.maximum(gap_array - tau, 0.0)
    return (base_array + beta * update).astype(np.float64).tolist()


def build_tau_grid(text_gaps: list[float], quantiles_arg: str) -> list[float]:
    gaps = np.asarray(text_gaps, dtype=np.float64)
    if gaps.size == 0:
        return [0.0]
    quantiles: list[float] = []
    for token in quantiles_arg.split(","):
        token = token.strip()
        if not token:
            continue
        quantiles.append(min(max(float(token), 0.0), 1.0))
    tau_values = [float(gaps.min() - 1e-6)]
    tau_values.extend(float(np.quantile(gaps, q)) for q in quantiles)
    return sorted({round(value, 10) for value in tau_values})


def build_beta_grid(beta_max: float, beta_steps: int) -> list[float]:
    if beta_steps <= 1:
        return [max(beta_max, 0.0)]
    return [float(value) for value in np.linspace(0.0, max(beta_max, 0.0), num=beta_steps)]


def select_split_parameters(
    support_train_records: list[dict[str, object]],
    holdout_records: list[dict[str, object]],
    text_gap_by_path: dict[str, float],
    beta_grid: list[float],
    tau_grid: list[float],
    bottle_drop_tolerance: float,
    min_delta: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    holdout_base_scores = [float(record["winner_image_score"]) for record in holdout_records]
    holdout_base_rows = evaluate_image_per_category(holdout_records, holdout_base_scores)
    baseline_summary = summarize_image_rows(holdout_base_rows)
    holdout_gaps = [text_gap_by_path[str(record["path"])] for record in holdout_records]

    best = {
        **baseline_summary,
        "beta": 0.0,
        "tau": float(tau_grid[0] if tau_grid else 0.0),
        "selection_metric": baseline_summary["weak5_image_auroc_mean"],
    }
    history: list[dict[str, float]] = []
    for beta in beta_grid:
        for tau in tau_grid:
            holdout_scores = apply_monotonic_update(holdout_base_scores, holdout_gaps, beta=beta, tau=tau)
            rows = evaluate_image_per_category(holdout_records, holdout_scores)
            summary = summarize_image_rows(rows)
            bottle_ok = summary["bottle_image_auroc"] >= baseline_summary["bottle_image_auroc"] - bottle_drop_tolerance
            candidate = {
                **summary,
                "beta": float(beta),
                "tau": float(tau),
                "bottle_ok": float(bottle_ok),
            }
            history.append(candidate)
            improved = (
                bottle_ok
                and (
                    summary["weak5_image_auroc_mean"] > float(best["selection_metric"]) + min_delta
                    or (
                        abs(summary["weak5_image_auroc_mean"] - float(best["selection_metric"])) <= min_delta
                        and summary["overall_image_auroc_mean"] > float(best["overall_image_auroc_mean"]) + min_delta
                    )
                )
            )
            if improved:
                best = {
                    **summary,
                    "beta": float(beta),
                    "tau": float(tau),
                    "selection_metric": summary["weak5_image_auroc_mean"],
                }
    best["support_train_records"] = float(len(support_train_records))
    best["holdout_records"] = float(len(holdout_records))
    return best, history


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, _ = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    all_records = train_records + eval_records
    categories = {str(record["category"]) for record in all_records}

    model = torch.jit.load(args.pretrained, map_location=device).eval()
    if device.type != "cpu":
        model = model.to(device)

    all_paths = [str(record["path"]) for record in all_records]
    image_features = encode_image_features(
        model=model,
        paths=all_paths,
        image_size=args.clip_image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    prompt_map = build_text_prompt_map(categories)
    text_prototypes = encode_text_prototypes(model=model, prompt_map=prompt_map, device=device)
    text_gap_by_path = compute_text_gaps(all_records, image_features=image_features, text_prototypes=text_prototypes)

    eval_base_scores = [float(record["winner_image_score"]) for record in eval_records]
    baseline_eval_rows = evaluate_image_per_category(eval_records, eval_base_scores)
    baseline_eval_summary = summarize_image_rows(baseline_eval_rows)

    beta_grid = build_beta_grid(args.beta_max, args.beta_steps)
    query_score_arrays: list[np.ndarray] = []
    split_summaries: list[dict[str, float]] = []
    search_history: list[dict[str, float]] = []

    for split_index in range(args.num_resplits):
        support_train_records, holdout_records = split_support_holdout(
            train_records=train_records,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed + split_index,
        )
        tau_grid = build_tau_grid(
            [text_gap_by_path[str(record["path"])] for record in support_train_records],
            quantiles_arg=args.tau_quantiles,
        )
        best_split, split_history = select_split_parameters(
            support_train_records=support_train_records,
            holdout_records=holdout_records,
            text_gap_by_path=text_gap_by_path,
            beta_grid=beta_grid,
            tau_grid=tau_grid,
            bottle_drop_tolerance=args.bottle_drop_tolerance,
            min_delta=args.min_delta,
        )
        for row in split_history:
            search_history.append({"split": float(split_index + 1), **row})
        split_summaries.append({"split": float(split_index + 1), **best_split})
        eval_gaps = [text_gap_by_path[str(record["path"])] for record in eval_records]
        split_scores = apply_monotonic_update(
            eval_base_scores,
            eval_gaps,
            beta=float(best_split["beta"]),
            tau=float(best_split["tau"]),
        )
        query_score_arrays.append(np.asarray(split_scores, dtype=np.float64))

    final_eval_scores = np.mean(np.stack(query_score_arrays, axis=0), axis=0).astype(np.float64).tolist()
    final_rows = evaluate_image_per_category(eval_records, final_eval_scores)
    final_summary = summarize_image_rows(final_rows)
    holdout_summary = {
        "overall_image_auroc_mean": float(np.mean([row["overall_image_auroc_mean"] for row in split_summaries])),
        "weak5_image_auroc_mean": float(np.mean([row["weak5_image_auroc_mean"] for row in split_summaries])),
        "bottle_image_auroc": float(np.mean([row["bottle_image_auroc"] for row in split_summaries])),
        "selection_beta_mean": float(np.mean([row["beta"] for row in split_summaries])),
        "selection_tau_mean": float(np.mean([row["tau"] for row in split_summaries])),
        "splits": split_summaries,
    }

    gate_pass_smoke = (
        final_summary["weak5_image_auroc_mean"] >= baseline_eval_summary["weak5_image_auroc_mean"] + args.weak5_min_delta
        and final_summary["bottle_image_auroc"] >= baseline_eval_summary["bottle_image_auroc"] - args.bottle_drop_tolerance
    )

    experiments_rows = [
        {
            "experiment": "E0",
            "num_categories": len(baseline_eval_rows),
            "image_auroc_mean": baseline_eval_summary["overall_image_auroc_mean"],
            "weak5_image_auroc_mean": baseline_eval_summary["weak5_image_auroc_mean"],
            "bottle_image_auroc": baseline_eval_summary["bottle_image_auroc"],
        },
        {
            "experiment": "E1-text-prior",
            "num_categories": len(final_rows),
            "image_auroc_mean": final_summary["overall_image_auroc_mean"],
            "weak5_image_auroc_mean": final_summary["weak5_image_auroc_mean"],
            "bottle_image_auroc": final_summary["bottle_image_auroc"],
            "holdout_overall_image_auroc_mean": holdout_summary["overall_image_auroc_mean"],
            "holdout_weak5_image_auroc_mean": holdout_summary["weak5_image_auroc_mean"],
            "holdout_bottle_image_auroc": holdout_summary["bottle_image_auroc"],
            "selection_beta_mean": holdout_summary["selection_beta_mean"],
            "selection_tau_mean": holdout_summary["selection_tau_mean"],
            "num_resplits": float(args.num_resplits),
            "gate_pass_smoke": bool(gate_pass_smoke),
        },
    ]

    per_category_rows = [{"experiment": "E0", **row} for row in baseline_eval_rows]
    per_category_rows.extend({"experiment": "E1-text-prior", **row} for row in final_rows)
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    (output_dir / "holdout_summary.json").write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")
    (output_dir / "search_history.json").write_text(json.dumps(search_history, indent=2), encoding="utf-8")

    text_prior_stats = {
        "pretrained": args.pretrained,
        "clip_image_size": args.clip_image_size,
        "num_resplits": args.num_resplits,
        "beta_grid": beta_grid,
        "tau_quantiles": [float(item.strip()) for item in args.tau_quantiles.split(",") if item.strip()],
        "prompt_map": prompt_map,
        "text_gap_summary": {
            category: {
                "mean": float(np.mean([text_gap_by_path[str(record["path"])] for record in all_records if str(record["category"]) == category])),
                "std": float(np.std([text_gap_by_path[str(record["path"])] for record in all_records if str(record["category"]) == category])),
            }
            for category in sorted(categories)
        },
        "split_summaries": split_summaries,
    }
    (output_dir / "text_prior_stats.json").write_text(json.dumps(text_prior_stats, indent=2), encoding="utf-8")

    summary_lines = [
        "# Stage3 text-prior summary",
        "",
        f"E0: image={format_metric(baseline_eval_summary['overall_image_auroc_mean'])}",
        f"E1-text-prior: image={format_metric(final_summary['overall_image_auroc_mean'])}",
        f"delta: image={final_summary['overall_image_auroc_mean'] - baseline_eval_summary['overall_image_auroc_mean']:+.4f}",
        (
            f"query gate: weak5_image={format_metric(final_summary['weak5_image_auroc_mean'])} "
            f"bottle_image={format_metric(final_summary['bottle_image_auroc'])} pass={gate_pass_smoke}"
        ),
        (
            f"holdout selector: weak5_image={format_metric(holdout_summary['weak5_image_auroc_mean'])} "
            f"bottle_image={format_metric(holdout_summary['bottle_image_auroc'])}"
        ),
        (
            f"selection: beta_mean={holdout_summary['selection_beta_mean']:.4f} "
            f"tau_mean={holdout_summary['selection_tau_mean']:.4f} resplits={args.num_resplits}"
        ),
        "note: final_score = E0 + beta * relu(text_gap - tau), using frozen image score plus frozen text gap only.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    result = {
        "mode": "p3_text_prior",
        "output_dir": str(output_dir),
        "baseline": experiments_rows[0],
        "candidate": experiments_rows[-1],
        "delta_image_auroc": final_summary["overall_image_auroc_mean"] - baseline_eval_summary["overall_image_auroc_mean"],
        "gate_pass_smoke": gate_pass_smoke,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
