import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.head import ImageResidualCalibrator
from fewshot.losses import pairwise_margin_rank
from run_stage3_head import append_jsonl, fit_image_score_calibration, load_records, resolve_device, set_seed, write_csv
from run_stage3_image_only import (
    build_anchor_selector,
    build_feature_tensor,
    build_label_tensor,
    evaluate_image_per_category,
    fit_hard_image_score_calibration,
    measure_identity_drift,
    measure_rank_activity,
    normalize_feature_splits,
    predict_scores,
    split_support_holdout,
    summarize_image_rows,
)


VARIANT_CURRENT = "current"
VARIANT_RANK_ONLY = "rank_only"
VARIANT_BCE_ONLY = "bce_only"
SELECTION_HOLDOUT = "holdout_selected"
SELECTION_QUERY = "query_best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 selector/objective diagnostic probes from cached Stage2 winner maps.")
    parser.add_argument("--cache-dir", default="outputs/stage3/cache/weak5_bottle/seed42")
    parser.add_argument("--output-dir", default="outputs/stage3/diagnostics/selector_objective/weak5_bottle/seed42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument("--hard-calibration-ratio", type=float, default=0.5)
    parser.add_argument("--anchor-normal-quantile", type=float, default=0.75)
    parser.add_argument("--gate-bias", type=float, default=-2.5)
    parser.add_argument("--rank-weight", type=float, default=0.6)
    parser.add_argument("--img-bce-weight", type=float, default=0.3)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--residual-l2-weight", type=float, default=0.05)
    parser.add_argument("--bottle-drop-tolerance", type=float, default=0.03)
    parser.add_argument("--variants", default="current,rank_only,bce_only")
    return parser.parse_args()


def parse_variants(raw: str) -> list[str]:
    allowed = {VARIANT_CURRENT, VARIANT_RANK_ONLY, VARIANT_BCE_ONLY}
    variants: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported variant '{token}', expected one of {sorted(allowed)}")
        variants.append(token)
    if not variants:
        raise ValueError("At least one variant is required.")
    return variants


def variant_loss(
    variant: str,
    rank: torch.Tensor,
    img_bce: torch.Tensor,
    anchor: torch.Tensor,
    residual_l2: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if variant == VARIANT_CURRENT:
        return (
            args.rank_weight * rank
            + args.img_bce_weight * img_bce
            + args.anchor_weight * anchor
            + args.residual_l2_weight * residual_l2
        )
    if variant == VARIANT_RANK_ONLY:
        return rank
    if variant == VARIANT_BCE_ONLY:
        return img_bce
    raise ValueError(f"Unknown variant: {variant}")


def is_better_holdout(
    candidate: dict[str, float],
    best: dict[str, float],
    candidate_rank_proxy: float,
    best_rank_proxy: float,
    bottle_drop_tolerance: float,
    min_delta: float,
) -> bool:
    if candidate["bottle_image_auroc"] < best["baseline_bottle_image_auroc"] - bottle_drop_tolerance:
        return False
    if candidate["weak5_image_auroc_mean"] > best["weak5_image_auroc_mean"] + min_delta:
        return True
    if abs(candidate["weak5_image_auroc_mean"] - best["weak5_image_auroc_mean"]) <= min_delta:
        if candidate_rank_proxy > best_rank_proxy + min_delta:
            return True
        if abs(candidate_rank_proxy - best_rank_proxy) <= min_delta:
            return candidate["overall_image_auroc_mean"] > best["overall_image_auroc_mean"] + min_delta
    return False


def is_better_query(candidate: dict[str, float], best: dict[str, float], min_delta: float) -> bool:
    if candidate["overall_image_auroc_mean"] > best["overall_image_auroc_mean"] + min_delta:
        return True
    if abs(candidate["overall_image_auroc_mean"] - best["overall_image_auroc_mean"]) <= min_delta:
        if candidate["weak5_image_auroc_mean"] > best["weak5_image_auroc_mean"] + min_delta:
            return True
        if abs(candidate["weak5_image_auroc_mean"] - best["weak5_image_auroc_mean"]) <= min_delta:
            return candidate["bottle_image_auroc"] > best["bottle_image_auroc"] + min_delta
    return False


def train_probe_variant(
    *,
    variant: str,
    model: ImageResidualCalibrator,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_frozen_logits: torch.Tensor,
    train_anchor_selector: torch.Tensor,
    holdout_records: list[dict[str, object]],
    holdout_x: torch.Tensor,
    holdout_frozen_logits: torch.Tensor,
    eval_records: list[dict[str, object]],
    eval_x: torch.Tensor,
    eval_frozen_logits: torch.Tensor,
    baseline_holdout_summary: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    split_index: int,
    metrics_path: Path | None = None,
) -> tuple[list[dict[str, float]], dict[str, float], list[float], dict[str, float], list[float]]:
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_frozen_logits, train_anchor_selector),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = float(train_y.sum().item())
    negatives = float(train_y.numel() - positives)
    pos_weight_value = max(1.0, negatives / positives) if positives > 0.0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    best_holdout = {
        **baseline_holdout_summary,
        "selection_epoch": 0.0,
        "selection_metric": baseline_holdout_summary["weak5_image_auroc_mean"],
        "baseline_bottle_image_auroc": baseline_holdout_summary["bottle_image_auroc"],
    }
    best_query = {
        "overall_image_auroc_mean": float("-inf"),
        "weak5_image_auroc_mean": float("-inf"),
        "bottle_image_auroc": float("-inf"),
        "selection_epoch": 0.0,
    }
    best_holdout_rank_proxy = float("-inf")
    best_holdout_scores: list[float] | None = None
    best_query_scores: list[float] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "img_bce": 0.0,
            "rank": 0.0,
            "anchor": 0.0,
            "residual_l2": 0.0,
            "gate_mean": 0.0,
            "residual_abs_mean": 0.0,
        }
        steps = 0

        for batch_x, batch_y, batch_frozen_logits, batch_anchor in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_frozen_logits = batch_frozen_logits.to(device)
            batch_anchor = batch_anchor.to(device)

            outputs = model(batch_x, batch_frozen_logits, return_dict=True)
            final_logits = outputs["final_logits"]
            residual = outputs["residual"]
            gate = outputs["gate"]

            img_bce = F.binary_cross_entropy_with_logits(final_logits, batch_y, pos_weight=pos_weight)
            rank = pairwise_margin_rank(final_logits, batch_y, margin=args.margin, smooth=True)
            if bool(batch_anchor.any().item()):
                anchor = F.mse_loss(final_logits[batch_anchor], batch_frozen_logits[batch_anchor])
            else:
                anchor = final_logits.new_zeros(())
            residual_l2 = residual.square().mean()
            loss = variant_loss(variant=variant, rank=rank, img_bce=img_bce, anchor=anchor, residual_l2=residual_l2, args=args)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sums["loss"] += float(loss.item())
            sums["img_bce"] += float(img_bce.item())
            sums["rank"] += float(rank.item())
            sums["anchor"] += float(anchor.item())
            sums["residual_l2"] += float(residual_l2.item())
            sums["gate_mean"] += float(gate.mean().item())
            sums["residual_abs_mean"] += float(residual.abs().mean().item())
            steps += 1

        holdout_scores, holdout_diag = predict_scores(
            model=model,
            features=holdout_x,
            frozen_logits=holdout_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )
        holdout_rows = evaluate_image_per_category(holdout_records, holdout_scores)
        holdout_summary = summarize_image_rows(holdout_rows)
        holdout_rank_proxy = -float(
            pairwise_margin_rank(
                torch.tensor(holdout_scores, dtype=torch.float32, device=device),
                torch.tensor([float(record["label"]) for record in holdout_records], dtype=torch.float32, device=device),
                margin=args.margin,
                smooth=True,
            ).item()
        )

        eval_scores, eval_diag = predict_scores(
            model=model,
            features=eval_x,
            frozen_logits=eval_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )
        eval_rows = evaluate_image_per_category(eval_records, eval_scores)
        eval_summary = summarize_image_rows(eval_rows)

        if is_better_holdout(
            candidate=holdout_summary,
            best=best_holdout,
            candidate_rank_proxy=holdout_rank_proxy,
            best_rank_proxy=best_holdout_rank_proxy,
            bottle_drop_tolerance=args.bottle_drop_tolerance,
            min_delta=args.min_delta,
        ):
            best_holdout = {
                **holdout_summary,
                "selection_epoch": float(epoch),
                "selection_metric": holdout_summary["weak5_image_auroc_mean"],
                "baseline_bottle_image_auroc": baseline_holdout_summary["bottle_image_auroc"],
            }
            best_holdout_rank_proxy = holdout_rank_proxy
            best_holdout_scores = list(eval_scores)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if is_better_query(candidate=eval_summary, best=best_query, min_delta=args.min_delta):
            best_query = {**eval_summary, "selection_epoch": float(epoch)}
            best_query_scores = list(eval_scores)

        row = {
            "variant": variant,
            "split": float(split_index),
            "epoch": float(epoch),
            **{key: value / max(1, steps) for key, value in sums.items()},
            "holdout_overall_image_auroc_mean": holdout_summary["overall_image_auroc_mean"],
            "holdout_weak5_image_auroc_mean": holdout_summary["weak5_image_auroc_mean"],
            "holdout_bottle_image_auroc": holdout_summary["bottle_image_auroc"],
            "holdout_rank_proxy": holdout_rank_proxy,
            "holdout_gate_mean": holdout_diag["gate_mean"],
            "holdout_residual_abs_mean": holdout_diag["residual_abs_mean"],
            "eval_overall_image_auroc_mean": eval_summary["overall_image_auroc_mean"],
            "eval_weak5_image_auroc_mean": eval_summary["weak5_image_auroc_mean"],
            "eval_bottle_image_auroc": eval_summary["bottle_image_auroc"],
            "eval_gate_mean": eval_diag["gate_mean"],
            "eval_residual_abs_mean": eval_diag["residual_abs_mean"],
            "best_holdout_epoch": best_holdout["selection_epoch"],
            "best_query_epoch": best_query["selection_epoch"],
        }
        history.append(row)
        append_jsonl(metrics_path, row)
        if stale_epochs >= args.patience:
            break

    if best_holdout_scores is None:
        current_scores, _ = predict_scores(
            model=model,
            features=eval_x,
            frozen_logits=eval_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )
        best_holdout_scores = list(current_scores)
    if best_query_scores is None:
        best_query_scores = list(best_holdout_scores)
        if best_query["selection_epoch"] <= 0.0:
            best_query = {
                "overall_image_auroc_mean": best_holdout["overall_image_auroc_mean"],
                "weak5_image_auroc_mean": best_holdout["weak5_image_auroc_mean"],
                "bottle_image_auroc": best_holdout["bottle_image_auroc"],
                "selection_epoch": best_holdout["selection_epoch"],
            }

    return history, best_holdout, best_holdout_scores, best_query, best_query_scores


def summarize_split_metric(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    return {
        "selection_epoch_mean": float(np.mean([row[f"{prefix}_selection_epoch"] for row in rows])),
    }


def write_summary(path: Path, baseline_summary: dict[str, float], rows: list[dict[str, object]]) -> None:
    lines = [
        "# Stage3 selector/objective probe summary",
        "",
        f"E0: image={baseline_summary['overall_image_auroc_mean']:.4f} weak5={baseline_summary['weak5_image_auroc_mean']:.4f} bottle={baseline_summary['bottle_image_auroc']:.4f}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} / {row['selection_mode']}: "
            f"image={float(row['image_auroc_mean']):.4f} "
            f"weak5={float(row['weak5_image_auroc_mean']):.4f} "
            f"bottle={float(row['bottle_image_auroc']):.4f} "
            f"epoch_mean={float(row['selection_epoch_mean']):.2f}"
        )
    lines.append("")
    lines.append("note: query_best is oracle diagnostic only and must not be treated as an official gate or retained winner.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    records, _ = load_records(cache_dir)
    train_records = [record for record in records if record["role"] in ("support_normal", "support_defect")]
    eval_records = [record for record in records if record["role"] == "query_eval"]
    baseline_eval_scores = [float(record["winner_image_score"]) for record in eval_records]
    baseline_eval_rows = evaluate_image_per_category(eval_records, baseline_eval_scores)
    baseline_eval_summary = summarize_image_rows(baseline_eval_rows)

    variants = parse_variants(args.variants)
    diagnostic_history: list[dict[str, float]] = []
    split_summary_rows: list[dict[str, float]] = []
    experiments_rows: list[dict[str, object]] = [
        {
            "variant": "baseline",
            "selection_mode": "frozen",
            "image_auroc_mean": baseline_eval_summary["overall_image_auroc_mean"],
            "weak5_image_auroc_mean": baseline_eval_summary["weak5_image_auroc_mean"],
            "bottle_image_auroc": baseline_eval_summary["bottle_image_auroc"],
            "selection_epoch_mean": 0.0,
        }
    ]

    for variant in variants:
        holdout_selected_score_arrays: list[np.ndarray] = []
        query_best_score_arrays: list[np.ndarray] = []
        variant_split_rows: list[dict[str, float]] = []
        for split_index in range(args.num_resplits):
            support_train_records, holdout_records = split_support_holdout(
                train_records=train_records,
                holdout_fraction=args.holdout_fraction,
                seed=args.seed + split_index,
            )
            train_y = build_label_tensor(support_train_records)
            support_train_scores = torch.tensor(
                [float(record["winner_image_score"]) for record in support_train_records],
                dtype=torch.float32,
            )
            image_calibration = fit_image_score_calibration(support_train_scores, train_y)
            hard_image_calibration = fit_hard_image_score_calibration(
                scores=support_train_scores,
                labels=train_y,
                base_calibration=image_calibration,
                hard_ratio=args.hard_calibration_ratio,
            )
            train_x_raw, train_frozen_logits, _ = build_feature_tensor(
                support_train_records,
                base_calibration=image_calibration,
                hard_calibration=hard_image_calibration,
            )
            holdout_x_raw, holdout_frozen_logits, _ = build_feature_tensor(
                holdout_records,
                base_calibration=image_calibration,
                hard_calibration=hard_image_calibration,
            )
            eval_x_raw, eval_frozen_logits, _ = build_feature_tensor(
                eval_records,
                base_calibration=image_calibration,
                hard_calibration=hard_image_calibration,
            )
            train_x, holdout_x, eval_x, _ = normalize_feature_splits(train_x_raw, holdout_x_raw, eval_x_raw)
            train_anchor_selector = build_anchor_selector(
                support_train_records,
                frozen_logits=train_frozen_logits,
                normal_quantile=args.anchor_normal_quantile,
            )
            baseline_holdout_scores = torch.sigmoid(holdout_frozen_logits).cpu().tolist()
            baseline_holdout_rows = evaluate_image_per_category(holdout_records, baseline_holdout_scores)
            baseline_holdout_summary = summarize_image_rows(baseline_holdout_rows)

            model = ImageResidualCalibrator(in_features=int(train_x.shape[1]), gate_bias=args.gate_bias).to(device)
            identity_check = measure_identity_drift(
                model=model,
                features=eval_x,
                frozen_logits=eval_frozen_logits,
                batch_size=args.batch_size,
                device=device,
            )
            if identity_check["max_abs_image_diff"] > 1e-6:
                raise RuntimeError(f"Identity drift before training for variant={variant} split={split_index + 1}: {identity_check}")
            rank_check = measure_rank_activity(
                model=model,
                features=train_x,
                frozen_logits=train_frozen_logits,
                labels=train_y,
                batch_size=args.batch_size,
                margin=args.margin,
                device=device,
            )
            if rank_check["rank_smooth"] <= 0.0:
                raise RuntimeError(f"Inactive rank objective for variant={variant} split={split_index + 1}: {rank_check}")

            split_history, best_holdout, best_holdout_scores, best_query, best_query_scores = train_probe_variant(
                variant=variant,
                model=model,
                train_x=train_x,
                train_y=train_y,
                train_frozen_logits=train_frozen_logits,
                train_anchor_selector=train_anchor_selector,
                holdout_records=holdout_records,
                holdout_x=holdout_x,
                holdout_frozen_logits=holdout_frozen_logits,
                eval_records=eval_records,
                eval_x=eval_x,
                eval_frozen_logits=eval_frozen_logits,
                baseline_holdout_summary=baseline_holdout_summary,
                args=args,
                device=device,
                split_index=split_index + 1,
                metrics_path=metrics_path,
            )
            diagnostic_history.extend(split_history)
            holdout_selected_score_arrays.append(np.asarray(best_holdout_scores, dtype=np.float64))
            query_best_score_arrays.append(np.asarray(best_query_scores, dtype=np.float64))
            variant_split_rows.append(
                {
                    "variant": variant,
                    "split": float(split_index + 1),
                    "holdout_selected_selection_epoch": best_holdout["selection_epoch"],
                    "query_best_selection_epoch": best_query["selection_epoch"],
                }
            )

        split_summary_rows.extend(variant_split_rows)
        for selection_mode, score_arrays in (
            (SELECTION_HOLDOUT, holdout_selected_score_arrays),
            (SELECTION_QUERY, query_best_score_arrays),
        ):
            final_scores = np.mean(np.stack(score_arrays, axis=0), axis=0).astype(np.float64).tolist()
            final_rows = evaluate_image_per_category(eval_records, final_scores)
            final_summary = summarize_image_rows(final_rows)
            prefix = "holdout_selected" if selection_mode == SELECTION_HOLDOUT else "query_best"
            selection_stats = summarize_split_metric(variant_split_rows, prefix)
            experiments_rows.append(
                {
                    "variant": variant,
                    "selection_mode": selection_mode,
                    "image_auroc_mean": final_summary["overall_image_auroc_mean"],
                    "weak5_image_auroc_mean": final_summary["weak5_image_auroc_mean"],
                    "bottle_image_auroc": final_summary["bottle_image_auroc"],
                    "selection_epoch_mean": selection_stats["selection_epoch_mean"],
                }
            )

    write_csv(output_dir / "experiments.csv", experiments_rows)
    (output_dir / "diagnostic_history.json").write_text(json.dumps(diagnostic_history, indent=2), encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary_rows, indent=2), encoding="utf-8")
    write_summary(output_dir / "summary.md", baseline_eval_summary, experiments_rows[1:])

    print(json.dumps({"mode": "selector_objective_probe", "output_dir": str(output_dir), "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
