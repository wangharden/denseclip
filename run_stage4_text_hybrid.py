import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from run_prompt_defect_text import (
    FEATURE_SOURCE_CLIP_GLOBAL,
    append_result_rows,
    build_feature_matrix,
    build_predictions_rows,
    classification_metrics,
    build_prompt_bank,
    build_resplits,
    compute_scores_for_records,
    encode_clip_global_features,
    encode_prompt_bank,
    evaluate_rows,
    mean_score_lists,
    parse_feature_sources,
    set_seed,
    train_split_model,
)
from run_stage3_head import load_records, resolve_device, split_records, write_csv


CONTROL_CATEGORY = "bottle"
STRONG_CONTROL_CATEGORIES = ("bottle", "zipper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage4 text-hybrid anchor experiments.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", default="")
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument("--feature-sources", default=FEATURE_SOURCE_CLIP_GLOBAL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--p-l2", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-grid", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--tau-grid", default="0.0,0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--margin-grid", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--control-scale-grid", default="0.25,0.5,0.75")
    parser.add_argument(
        "--baseline-experiments",
        default="",
        help="Optional experiments.csv for the current Stage4 prompt-text incumbent.",
    )
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def parse_float_grid(value: str) -> list[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def choose_threshold_dense(labels: list[int], scores: list[float]) -> dict[str, float]:
    values = sorted({float(score) for score in scores})
    if not values:
        return {"threshold": 0.0, "balanced_accuracy": 0.0, "f1": 0.0}
    candidates = [values[0] - 1e-6]
    for index, value in enumerate(values):
        candidates.append(value)
        if index + 1 < len(values):
            candidates.append(0.5 * (value + values[index + 1]))
    candidates.append(values[-1] + 1e-6)

    best = {
        "threshold": candidates[0],
        "balanced_accuracy": -1.0,
        "f1": -1.0,
        "accuracy": -1.0,
        "support_margin": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }
    for threshold in candidates:
        metrics = classification_metrics(labels=labels, scores=scores, threshold=float(threshold))
        support_margin = min(abs(float(threshold) - value) for value in values)
        candidate_key = (
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["accuracy"],
            support_margin,
        )
        best_key = (
            best["balanced_accuracy"],
            best["f1"],
            best["accuracy"],
            best["support_margin"],
        )
        if candidate_key > best_key:
            best = {"threshold": float(threshold), "support_margin": float(support_margin), **metrics}
    return best


def choose_thresholds_by_category_dense(
    records: list[dict[str, object]],
    scores: list[float],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        payload = grouped.setdefault(category, {"labels": [], "scores": []})
        payload["labels"].append(int(record["label"]))
        payload["scores"].append(float(score))
    threshold_map: dict[str, dict[str, float]] = {}
    for category, payload in sorted(grouped.items()):
        threshold_map[category] = choose_threshold_dense(payload["labels"], payload["scores"])
    return threshold_map


def load_incumbent_metrics(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    for row in rows:
        if row.get("experiment") == "prompt_plus_p__clip_global":
            return {
                "source_path": str(path),
                "experiment": row["experiment"],
                "image_auroc_mean": float(row["image_auroc_mean"]),
                "image_ap_mean": float(row["image_ap_mean"]),
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "specificity": float(row["specificity"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "f1": float(row["f1"]),
                "bottle_image_auroc": float(row["bottle_image_auroc"]),
                "zipper_image_auroc": float(row.get("zipper_image_auroc", "nan")),
            }
    raise ValueError(f"Could not find prompt_plus_p__clip_global in {path}")


def infer_incumbent_path(cache_dir: Path) -> Path:
    scope = cache_dir.parent.name
    if scope == "full15":
        return Path("outputs/new-branch/prompt-text/full15/seed42/prompt_p_v3_full15/experiments.csv")
    return Path("outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_p_v3_thresholdfix/experiments.csv")


def pooled_holdout_records(resplits: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for resplit in resplits:
        records.extend(resplit["holdout"])
    return records


def evaluate_rows_with_controls(
    records: list[dict[str, object]],
    scores: list[float],
    threshold_map: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    per_category, aggregate = evaluate_rows(records=records, scores=scores, threshold_map=threshold_map)
    by_category = {row["category"]: row for row in per_category}
    aggregate["bottle_image_auroc"] = float(by_category[CONTROL_CATEGORY]["image_auroc"])
    aggregate["zipper_image_auroc"] = float(by_category["zipper"]["image_auroc"]) if "zipper" in by_category else float("nan")
    return per_category, aggregate


def build_category_normal_stats(
    records: list[dict[str, object]],
    scores: list[float],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for record, score in zip(records, scores, strict=True):
        if int(record["label"]) != 0:
            continue
        grouped.setdefault(str(record["category"]), []).append(float(score))
    stats: dict[str, dict[str, float]] = {}
    for category, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        std = float(arr.std())
        stats[category] = {
            "mean": float(arr.mean()),
            "std": 1.0 if std < 1e-6 else std,
            "count": float(arr.size),
        }
    return stats


def build_category_quantile_stats(
    records: list[dict[str, object]],
    scores: list[float],
    quantile: float,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for record, score in zip(records, scores, strict=True):
        if int(record["label"]) != 0:
            continue
        grouped.setdefault(str(record["category"]), []).append(float(score))
    stats: dict[str, dict[str, float]] = {}
    for category, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        center = float(np.quantile(arr, quantile))
        median = float(np.median(arr))
        std = float(arr.std())
        scale = max(abs(center - median), std, 1e-6)
        stats[category] = {
            "mean": center,
            "std": scale,
            "count": float(arr.size),
            "median": median,
            "quantile": quantile,
        }
    return stats


def normalize_prompt_scores(
    records: list[dict[str, object]],
    scores: list[float],
    stats: dict[str, dict[str, float]],
) -> list[float]:
    normalized: list[float] = []
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        payload = stats[category]
        normalized.append((float(score) - float(payload["mean"])) / float(payload["std"]))
    return normalized


def fuse_scores(
    records: list[dict[str, object]],
    base_scores: list[float],
    prompt_z_scores: list[float],
    base_threshold_map: dict[str, dict[str, float]],
    lam: float,
    tau: float,
    margin: float | None,
    control_scale: float,
) -> tuple[list[float], list[dict[str, object]]]:
    fused: list[float] = []
    details: list[dict[str, object]] = []
    for record, base_score, prompt_z in zip(records, base_scores, prompt_z_scores, strict=True):
        category = str(record["category"])
        base_threshold = float(base_threshold_map[category]["threshold"])
        residual = max(0.0, float(prompt_z) - tau)
        gate = 1.0
        if margin is not None:
            gate = 1.0 if abs(float(base_score) - base_threshold) <= margin else 0.0
        category_scale = control_scale if category in STRONG_CONTROL_CATEGORIES else 1.0
        delta = lam * gate * category_scale * residual
        fused_score = float(base_score) + delta
        fused.append(fused_score)
        details.append(
            {
                "base_score": float(base_score),
                "prompt_z": float(prompt_z),
                "base_threshold": base_threshold,
                "gate": float(gate),
                "delta": float(delta),
            }
        )
    return fused, details


def build_hybrid_prediction_rows(
    experiment: str,
    records: list[dict[str, object]],
    scores: list[float],
    base_details: list[dict[str, object]],
    hybrid_details: list[dict[str, object]],
    threshold_map: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows = build_predictions_rows(
        experiment=experiment,
        records=records,
        scores=scores,
        details=base_details,
        threshold_map=threshold_map,
    )
    for row, detail in zip(rows, hybrid_details, strict=True):
        row["base_score"] = detail["base_score"]
        row["prompt_z"] = detail["prompt_z"]
        row["base_threshold"] = detail["base_threshold"]
        row["gate"] = detail["gate"]
        row["delta"] = detail["delta"]
    return rows


def aggregate_prediction_health(predictions: list[dict[str, object]]) -> dict[str, float]:
    total = len(predictions)
    positives = int(sum(int(row["pred_label"]) for row in predictions))
    ratio = positives / max(1, total)
    return {
        "num_pred_positive": float(positives),
        "num_pred_negative": float(total - positives),
        "positive_ratio": float(ratio),
    }


def is_finite_aggregate(row: dict[str, object]) -> bool:
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def select_best_candidate(
    variant_name: str,
    pooled_records: list[dict[str, object]],
    pooled_base_scores: list[float],
    pooled_prompt_z_scores: list[float],
    eval_records: list[dict[str, object]],
    eval_base_scores: list[float],
    eval_prompt_z_scores: list[float],
    eval_details: list[dict[str, object]],
    base_threshold_map: dict[str, dict[str, float]],
    e0_holdout: dict[str, float],
    lambda_grid: list[float],
    tau_grid: list[float],
    margin_grid: list[float],
    control_scales: list[float],
    boundary_mode: bool,
    control_mode: bool,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    search_rows: list[dict[str, object]] = []

    margins = [None] if not boundary_mode else margin_grid
    control_values = [1.0] if not control_mode else control_scales
    pooled_by_category = {row["category"] for row in pooled_records}

    for lam in lambda_grid:
        for tau in tau_grid:
            for margin in margins:
                for control_scale in control_values:
                    holdout_scores, _ = fuse_scores(
                        records=pooled_records,
                        base_scores=pooled_base_scores,
                        prompt_z_scores=pooled_prompt_z_scores,
                        base_threshold_map=base_threshold_map,
                        lam=lam,
                        tau=tau,
                        margin=margin,
                        control_scale=control_scale,
                    )
                    threshold_map = choose_thresholds_by_category_dense(records=pooled_records, scores=holdout_scores)
                    _, holdout_aggregate = evaluate_rows_with_controls(
                        records=pooled_records,
                        scores=holdout_scores,
                        threshold_map=threshold_map,
                    )
                    health = aggregate_prediction_health(
                        build_predictions_rows(
                            experiment=variant_name,
                            records=pooled_records,
                            scores=holdout_scores,
                            details=[{} for _ in pooled_records],
                            threshold_map=threshold_map,
                        )
                    )
                    bottle_ok = holdout_aggregate["bottle_image_auroc"] >= e0_holdout["bottle_image_auroc"] - 0.02
                    zipper_ok = True
                    if "zipper" in pooled_by_category and math.isfinite(e0_holdout["zipper_image_auroc"]):
                        zipper_ok = holdout_aggregate["zipper_image_auroc"] >= e0_holdout["zipper_image_auroc"] - 0.02
                    collapse = health["positive_ratio"] <= 0.01 or health["positive_ratio"] >= 0.99
                    candidate = {
                        "variant": variant_name,
                        "lambda": lam,
                        "tau": tau,
                        "margin": margin,
                        "control_scale": control_scale,
                        "holdout_image_auroc_mean": holdout_aggregate["image_auroc_mean"],
                        "holdout_image_ap_mean": holdout_aggregate["image_ap_mean"],
                        "holdout_balanced_accuracy": holdout_aggregate["balanced_accuracy"],
                        "holdout_bottle_image_auroc": holdout_aggregate["bottle_image_auroc"],
                        "holdout_zipper_image_auroc": holdout_aggregate["zipper_image_auroc"],
                        "holdout_positive_ratio": health["positive_ratio"],
                        "bottle_constraint_ok": bottle_ok,
                        "zipper_constraint_ok": zipper_ok,
                        "prediction_collapse": collapse,
                        "mean_delta_abs": float(
                            np.mean(
                                [
                                    abs(float(holdout_score) - float(base_score))
                                    for holdout_score, base_score in zip(holdout_scores, pooled_base_scores, strict=True)
                                ]
                            )
                        ),
                        "control_mean_delta_abs": float(
                            np.mean(
                                [
                                    abs(float(holdout_score) - float(base_score))
                                    for record, holdout_score, base_score in zip(
                                        pooled_records,
                                        holdout_scores,
                                        pooled_base_scores,
                                        strict=True,
                                    )
                                    if str(record["category"]) in STRONG_CONTROL_CATEGORIES
                                ]
                                or [0.0]
                            )
                        ),
                    }
                    search_rows.append(candidate)
                    sort_key = (
                        int(bottle_ok and zipper_ok and not collapse),
                        holdout_aggregate["image_auroc_mean"],
                        holdout_aggregate["image_ap_mean"],
                        holdout_aggregate["balanced_accuracy"],
                        holdout_aggregate["bottle_image_auroc"],
                        -candidate["control_mean_delta_abs"],
                        -candidate["mean_delta_abs"],
                    )
                    if best is None or sort_key > best["sort_key"]:
                        best = {
                            "sort_key": sort_key,
                            "lambda": lam,
                            "tau": tau,
                            "margin": margin,
                            "control_scale": control_scale,
                            "holdout_aggregate": holdout_aggregate,
                            "threshold_map": threshold_map,
                        }

    if best is None:
        raise RuntimeError(f"No candidate was produced for {variant_name}")

    eval_scores, hybrid_details = fuse_scores(
        records=eval_records,
        base_scores=eval_base_scores,
        prompt_z_scores=eval_prompt_z_scores,
        base_threshold_map=base_threshold_map,
        lam=float(best["lambda"]),
        tau=float(best["tau"]),
        margin=float(best["margin"]) if best["margin"] is not None else None,
        control_scale=float(best["control_scale"]),
    )
    per_category, aggregate = evaluate_rows_with_controls(
        records=eval_records,
        scores=eval_scores,
        threshold_map=best["threshold_map"],
    )
    predictions = build_hybrid_prediction_rows(
        experiment=variant_name,
        records=eval_records,
        scores=eval_scores,
        base_details=eval_details,
        hybrid_details=hybrid_details,
        threshold_map=best["threshold_map"],
    )
    return {
        "experiment": variant_name,
        "selection": {
            "lambda": float(best["lambda"]),
            "tau": float(best["tau"]),
            "margin": None if best["margin"] is None else float(best["margin"]),
            "control_scale": float(best["control_scale"]),
            "holdout_aggregate": best["holdout_aggregate"],
        },
        "aggregate": aggregate,
        "per_category": per_category,
        "predictions": predictions,
        "search_rows": search_rows,
    }


def write_summary(
    path: Path,
    experiments_rows: list[dict[str, object]],
    incumbent_metrics: dict[str, object],
) -> None:
    lines = ["# Stage4 Text Hybrid Summary", ""]
    for row in experiments_rows:
        lines.append(
            f"{row['experiment']}: auroc={row['image_auroc_mean']:.6f} ap={row['image_ap_mean']:.6f} "
            f"acc={row['accuracy']:.6f} bal_acc={row['balanced_accuracy']:.6f} "
            f"bottle={row['bottle_image_auroc']:.6f} zipper={row['zipper_image_auroc']:.6f}"
        )
    e0 = experiments_rows[0]
    lines.append("")
    lines.append(
        "incumbent_prompt_text: "
        f"auroc={incumbent_metrics['image_auroc_mean']:.6f} ap={incumbent_metrics['image_ap_mean']:.6f} "
        f"acc={incumbent_metrics['accuracy']:.6f} bal_acc={incumbent_metrics['balanced_accuracy']:.6f} "
        f"bottle={incumbent_metrics['bottle_image_auroc']:.6f} zipper={incumbent_metrics['zipper_image_auroc']:.6f}"
    )
    for row in experiments_rows[1:]:
        lines.append(
            f"delta_vs_E0 {row['experiment']}: "
            f"auroc={row['image_auroc_mean'] - e0['image_auroc_mean']:+.6f} "
            f"ap={row['image_ap_mean'] - e0['image_ap_mean']:+.6f} "
            f"acc={row['accuracy'] - e0['accuracy']:+.6f} "
            f"bal_acc={row['balanced_accuracy'] - e0['balanced_accuracy']:+.6f} "
            f"bottle={row['bottle_image_auroc'] - e0['bottle_image_auroc']:+.6f} "
            f"zipper={row['zipper_image_auroc'] - e0['zipper_image_auroc']:+.6f}"
        )
        lines.append(
            f"delta_vs_incumbent {row['experiment']}: "
            f"auroc={row['image_auroc_mean'] - incumbent_metrics['image_auroc_mean']:+.6f} "
            f"ap={row['image_ap_mean'] - incumbent_metrics['image_ap_mean']:+.6f} "
            f"acc={row['accuracy'] - incumbent_metrics['accuracy']:+.6f} "
            f"bal_acc={row['balanced_accuracy'] - incumbent_metrics['balanced_accuracy']:+.6f} "
            f"bottle={row['bottle_image_auroc'] - incumbent_metrics['bottle_image_auroc']:+.6f} "
            f"zipper={row['zipper_image_auroc'] - incumbent_metrics['zipper_image_auroc']:+.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    start_time = time.time()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    scope = args.scope.strip() or cache_dir.parent.name or cache_dir.name
    feature_sources = parse_feature_sources(args.feature_sources)
    if feature_sources != [FEATURE_SOURCE_CLIP_GLOBAL]:
        raise ValueError("Stage4 text-hybrid currently supports only clip_global to preserve the retained visual source.")

    incumbent_path = Path(args.baseline_experiments) if args.baseline_experiments else infer_incumbent_path(cache_dir)
    incumbent_metrics = load_incumbent_metrics(incumbent_path)

    records, _ = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    resplits = build_resplits(
        train_records=train_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        num_resplits=args.num_resplits,
    )
    pooled_holdout = pooled_holdout_records(resplits)

    clip_model = torch.jit.load(args.pretrained, map_location=device).eval()
    if device.type == "cuda":
        clip_model = clip_model.to(device)

    all_paths = sorted({str(record["path"]) for record in records})
    clip_features = encode_clip_global_features(
        model=clip_model,
        paths=all_paths,
        image_size=args.clip_image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    prompt_bank = build_prompt_bank(records)
    prompt_embeddings = encode_prompt_bank(model=clip_model, prompt_bank=prompt_bank, device=device)

    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    config = {
        "track": "new-branch/text-hybrid",
        "scope": scope,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "num_resplits": args.num_resplits,
        "holdout_fraction": args.holdout_fraction,
        "feature_sources": feature_sources,
        "pretrained": args.pretrained,
        "selection_contract": "pooled_holdout_prompt_normalization + pooled_holdout_threshold_search + frozen_stage2_anchor",
        "prompt_incumbent_reference": str(incumbent_path),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    baseline_holdout_scores = [float(record["winner_image_score"]) for record in pooled_holdout]
    baseline_threshold_map = choose_thresholds_by_category_dense(records=pooled_holdout, scores=baseline_holdout_scores)
    e0_holdout_per_category, e0_holdout_aggregate = evaluate_rows_with_controls(
        records=pooled_holdout,
        scores=baseline_holdout_scores,
        threshold_map=baseline_threshold_map,
    )
    baseline_eval_scores = [float(record["winner_image_score"]) for record in eval_records]
    e0_per_category, e0_aggregate = evaluate_rows_with_controls(
        records=eval_records,
        scores=baseline_eval_scores,
        threshold_map=baseline_threshold_map,
    )
    e0_predictions = build_predictions_rows(
        experiment="E0",
        records=eval_records,
        scores=baseline_eval_scores,
        details=[{} for _ in eval_records],
        threshold_map=baseline_threshold_map,
    )

    experiments_rows: list[dict[str, object]] = []
    per_category_rows: list[dict[str, object]] = []
    predictions_rows: list[dict[str, object]] = []
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
        threshold_source="per_category_pooled_multisplit_holdout_midgap_balanced_accuracy",
        threshold_value="per_category",
        scope=scope,
    )
    experiments_rows[-1]["zipper_image_auroc"] = e0_aggregate["zipper_image_auroc"]

    frozen_holdout_scores, _ = compute_scores_for_records(
        features_by_path=clip_features,
        records=pooled_holdout,
        prompt_embeddings=prompt_embeddings,
        p_vector=None,
        device=device,
    )
    frozen_eval_scores, frozen_eval_details = compute_scores_for_records(
        features_by_path=clip_features,
        records=eval_records,
        prompt_embeddings=prompt_embeddings,
        p_vector=None,
        device=device,
    )
    frozen_stats = build_category_normal_stats(records=pooled_holdout, scores=frozen_holdout_scores)
    frozen_holdout_z = normalize_prompt_scores(records=pooled_holdout, scores=frozen_holdout_scores, stats=frozen_stats)
    frozen_eval_z = normalize_prompt_scores(records=eval_records, scores=frozen_eval_scores, stats=frozen_stats)

    support_normal_records = [record for record in train_records if int(record["label"]) == 0]
    support_normal_scores, _ = compute_scores_for_records(
        features_by_path=clip_features,
        records=support_normal_records,
        prompt_embeddings=prompt_embeddings,
        p_vector=None,
        device=device,
    )
    frozen_support_stats = build_category_normal_stats(records=support_normal_records, scores=support_normal_scores)
    frozen_support_holdout_z = normalize_prompt_scores(records=pooled_holdout, scores=frozen_holdout_scores, stats=frozen_support_stats)
    frozen_support_eval_z = normalize_prompt_scores(records=eval_records, scores=frozen_eval_scores, stats=frozen_support_stats)
    frozen_support_q90_stats = build_category_quantile_stats(records=support_normal_records, scores=support_normal_scores, quantile=0.9)
    frozen_support_q90_holdout_z = normalize_prompt_scores(
        records=pooled_holdout,
        scores=frozen_holdout_scores,
        stats=frozen_support_q90_stats,
    )
    frozen_support_q90_eval_z = normalize_prompt_scores(
        records=eval_records,
        scores=frozen_eval_scores,
        stats=frozen_support_q90_stats,
    )

    split_eval_scores: list[list[float]] = []
    split_eval_details: list[list[dict[str, object]]] = []
    pooled_plus_holdout_scores: list[float] = []
    training_summary: list[dict[str, object]] = []
    for resplit in resplits:
        support_train = resplit["support_train"]
        holdout = resplit["holdout"]
        train_features = build_feature_matrix(support_train, clip_features)
        train_labels = torch.tensor([float(record["label"]) for record in support_train], dtype=torch.float32)
        train_categories = [str(record["category"]) for record in support_train]
        holdout_features = build_feature_matrix(holdout, clip_features)
        holdout_labels = [int(record["label"]) for record in holdout]
        holdout_categories = [str(record["category"]) for record in holdout]
        split_result = train_split_model(
            split_index=int(resplit["split_index"]),
            source_name=FEATURE_SOURCE_CLIP_GLOBAL,
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
        holdout_scores, _ = compute_scores_for_records(
            features_by_path=clip_features,
            records=holdout,
            prompt_embeddings=prompt_embeddings,
            p_vector=best_state,
            device=device,
        )
        eval_scores, eval_details = compute_scores_for_records(
            features_by_path=clip_features,
            records=eval_records,
            prompt_embeddings=prompt_embeddings,
            p_vector=best_state,
            device=device,
        )
        pooled_plus_holdout_scores.extend(holdout_scores)
        split_eval_scores.append(eval_scores)
        split_eval_details.append(eval_details)
        training_summary.append(
            {
                "split_index": int(resplit["split_index"]),
                "split_seed": int(resplit["split_seed"]),
                "best_epoch": int(split_result["best_epoch"]),
                "best_holdout_metric": float(split_result["best_holdout_metric"]),
            }
        )

    plus_eval_scores = mean_score_lists(split_eval_scores)
    plus_eval_details = split_eval_details[0] if split_eval_details else [{} for _ in eval_records]
    plus_stats = build_category_normal_stats(records=pooled_holdout, scores=pooled_plus_holdout_scores)
    plus_holdout_z = normalize_prompt_scores(records=pooled_holdout, scores=pooled_plus_holdout_scores, stats=plus_stats)
    plus_eval_z = normalize_prompt_scores(records=eval_records, scores=plus_eval_scores, stats=plus_stats)

    candidate_search: dict[str, object] = {
        "e0_holdout_aggregate": e0_holdout_aggregate,
        "training_summary": training_summary,
        "variants": {},
    }
    alerts: list[dict[str, object]] = []

    lambda_grid = parse_float_grid(args.lambda_grid)
    tau_grid = parse_float_grid(args.tau_grid)
    margin_grid = parse_float_grid(args.margin_grid)
    control_scales = parse_float_grid(args.control_scale_grid)

    variants = [
        ("hybrid_frozen__relu_z", frozen_holdout_z, frozen_eval_z, frozen_eval_details, False, False),
        ("hybrid_frozen__control_relu_z", frozen_holdout_z, frozen_eval_z, frozen_eval_details, False, True),
        ("hybrid_frozen__support_z_relu", frozen_support_holdout_z, frozen_support_eval_z, frozen_eval_details, False, False),
        ("hybrid_frozen__support_q90_relu", frozen_support_q90_holdout_z, frozen_support_q90_eval_z, frozen_eval_details, False, False),
        ("hybrid_plus_p__relu_z", plus_holdout_z, plus_eval_z, plus_eval_details, False, False),
        ("hybrid_plus_p__control_relu_z", plus_holdout_z, plus_eval_z, plus_eval_details, False, True),
        ("hybrid_plus_p__boundary_relu_z", plus_holdout_z, plus_eval_z, plus_eval_details, True, False),
        ("hybrid_plus_p__boundary_control_relu_z", plus_holdout_z, plus_eval_z, plus_eval_details, True, True),
    ]

    for variant_name, holdout_prompt_z, eval_prompt_z, eval_details, boundary_mode, control_mode in variants:
        result = select_best_candidate(
            variant_name=variant_name,
            pooled_records=pooled_holdout,
            pooled_base_scores=baseline_holdout_scores,
            pooled_prompt_z_scores=holdout_prompt_z,
            eval_records=eval_records,
            eval_base_scores=baseline_eval_scores,
            eval_prompt_z_scores=eval_prompt_z,
            eval_details=eval_details,
            base_threshold_map=baseline_threshold_map,
            e0_holdout=e0_holdout_aggregate,
            lambda_grid=lambda_grid,
            tau_grid=tau_grid,
            margin_grid=margin_grid,
            control_scales=control_scales,
            boundary_mode=boundary_mode,
            control_mode=control_mode,
        )
        aggregate = result["aggregate"]
        if not is_finite_aggregate(aggregate):
            alerts.append({"variant": variant_name, "alert": "non_finite_aggregate"})
            continue
        health = aggregate_prediction_health(result["predictions"])
        if health["positive_ratio"] <= 0.01 or health["positive_ratio"] >= 0.99:
            alerts.append({"variant": variant_name, "alert": "degenerate_predictions", **health})
        if aggregate["bottle_image_auroc"] < e0_aggregate["bottle_image_auroc"] - 0.02:
            alerts.append(
                {
                    "variant": variant_name,
                    "alert": "bottle_regression",
                    "bottle_delta": aggregate["bottle_image_auroc"] - e0_aggregate["bottle_image_auroc"],
                }
            )
        append_result_rows(
            experiments_rows=experiments_rows,
            per_category_rows=per_category_rows,
            predictions_rows=predictions_rows,
            experiment_name=variant_name,
            seed=args.seed,
            aggregate=aggregate,
            per_category=result["per_category"],
            predictions=result["predictions"],
            selection_source="pooled_holdout_grid_search",
            threshold_source="per_category_pooled_multisplit_holdout_midgap_balanced_accuracy",
            threshold_value=json.dumps(result["selection"], ensure_ascii=True),
            scope=scope,
        )
        experiments_rows[-1]["zipper_image_auroc"] = aggregate["zipper_image_auroc"]
        candidate_search["variants"][variant_name] = {
            "selection": result["selection"],
            "search_rows": result["search_rows"],
        }

    elapsed = time.time() - start_time
    runtime_notes = {
        "elapsed_seconds": elapsed,
        "device": str(device),
        "num_records": len(records),
        "num_eval_records": len(eval_records),
        "num_holdout_records": len(pooled_holdout),
    }

    (output_dir / "baseline_refs.json").write_text(
        json.dumps(
            {
                "stage2_e0_from_this_run": e0_aggregate,
                "stage4_prompt_text_incumbent": incumbent_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "num_resplits": args.num_resplits,
                "training_summary": training_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "prompt_normalization.json").write_text(
        json.dumps(
            {
                "frozen_holdout_normal": frozen_stats,
                "frozen_support_normal": frozen_support_stats,
                "frozen_support_q90": frozen_support_q90_stats,
                "plus_p_holdout_normal": plus_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "candidate_search.json").write_text(json.dumps(candidate_search, indent=2), encoding="utf-8")
    (output_dir / "alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    (output_dir / "runtime_notes.json").write_text(json.dumps(runtime_notes, indent=2), encoding="utf-8")
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "predictions.csv", predictions_rows)
    write_summary(output_dir / "summary.md", experiments_rows=experiments_rows, incumbent_metrics=incumbent_metrics)


if __name__ == "__main__":
    main()
