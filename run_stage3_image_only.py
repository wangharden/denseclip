import argparse
import json
import math
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
from fewshot.stage_a1 import binary_auroc
from run_stage3_head import (
    aggregate_rows,
    append_jsonl,
    apply_score_calibration,
    evaluate_per_category,
    fit_image_score_calibration,
    fit_score_calibration,
    load_records,
    resolve_device,
    set_seed,
    split_records,
    write_csv,
)


MODE_P3 = "p3"
CONTROL_CATEGORY = "bottle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 image-only residual calibrator from cached Stage2 winner maps.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
    return parser.parse_args()


def build_residual_feature_vector(
    record: dict[str, object],
    *,
    base_logit: float,
    hard_logit: float,
) -> list[float]:
    hard_gap = hard_logit - base_logit
    return [
        *record["feature_vector"],
        float(record["winner_image_score"]),
        float(base_logit),
        float(hard_logit),
        float(hard_gap),
        float(abs(base_logit)),
    ]


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
        order = rng.permutation(len(group))
        holdout_ids = set(order[:holdout_count].tolist())
        for index, record in enumerate(group):
            if index in holdout_ids:
                holdout_split.append(record)
            else:
                train_split.append(record)

    if not train_split or not holdout_split:
        raise ValueError("Support holdout split produced an empty train or holdout partition.")
    return train_split, holdout_split


def normalize_feature_splits(
    train_x: torch.Tensor,
    *other_groups: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    outputs: list[torch.Tensor] = [(train_x - mean) / std]
    for group in other_groups:
        outputs.append((group - mean) / std)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    outputs.append(stats)  # type: ignore[arg-type]
    return tuple(outputs)


def build_label_tensor(records: list[dict[str, object]]) -> torch.Tensor:
    return torch.tensor([float(record["label"]) for record in records], dtype=torch.float32)


def build_frozen_logits(
    records: list[dict[str, object]],
    calibration: dict[str, float],
) -> torch.Tensor:
    scores = torch.tensor([float(record["winner_image_score"]) for record in records], dtype=torch.float32)
    return apply_score_calibration(scores, calibration)


def fit_hard_image_score_calibration(
    scores: torch.Tensor,
    labels: torch.Tensor,
    base_calibration: dict[str, float],
    hard_ratio: float,
) -> dict[str, float]:
    hard_ratio = min(max(hard_ratio, 0.1), 1.0)
    base_logits = apply_score_calibration(scores, base_calibration)
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    positive_scores = scores[pos_mask]
    negative_scores = scores[neg_mask]
    positive_logits = base_logits[pos_mask]
    negative_logits = base_logits[neg_mask]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return dict(base_calibration)
    hard_pos_count = min(positive_scores.numel(), max(1, int(math.ceil(positive_scores.numel() * hard_ratio))))
    hard_neg_count = min(negative_scores.numel(), max(1, int(math.ceil(negative_scores.numel() * hard_ratio))))
    hard_pos_ids = torch.argsort(positive_logits)[:hard_pos_count]
    hard_neg_ids = torch.argsort(negative_logits, descending=True)[:hard_neg_count]
    return fit_score_calibration(
        positive_values=positive_scores[hard_pos_ids],
        negative_values=negative_scores[hard_neg_ids],
    )


def build_feature_tensor(
    records: list[dict[str, object]],
    base_calibration: dict[str, float],
    hard_calibration: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_scores = torch.tensor([float(record["winner_image_score"]) for record in records], dtype=torch.float32)
    base_logits = apply_score_calibration(raw_scores, base_calibration)
    hard_logits = apply_score_calibration(raw_scores, hard_calibration)
    feature_rows = [
        build_residual_feature_vector(
            record,
            base_logit=float(base_logit.item()),
            hard_logit=float(hard_logit.item()),
        )
        for record, base_logit, hard_logit in zip(records, base_logits, hard_logits, strict=True)
    ]
    return torch.tensor(feature_rows, dtype=torch.float32), base_logits, hard_logits


def build_anchor_selector(
    records: list[dict[str, object]],
    frozen_logits: torch.Tensor,
    normal_quantile: float,
) -> torch.Tensor:
    normal_quantile = min(max(normal_quantile, 0.0), 1.0)
    normal_candidates = torch.tensor(
        [
            int(record["label"]) == 0 and str(record["category"]) != CONTROL_CATEGORY
            for record in records
        ],
        dtype=torch.bool,
    )
    threshold = float("inf")
    if bool(normal_candidates.any().item()):
        normal_confidence = frozen_logits[normal_candidates].abs().cpu().numpy()
        threshold = float(np.quantile(normal_confidence, normal_quantile))
    return torch.tensor(
        [
            str(record["category"]) == CONTROL_CATEGORY
            or (
                int(record["label"]) == 0
                and str(record["category"]) != CONTROL_CATEGORY
                and abs(float(logit.item())) >= threshold
            )
            for record, logit in zip(records, frozen_logits, strict=True)
        ],
        dtype=torch.bool,
    )


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


def _binary_pos_weight(labels: torch.Tensor) -> float:
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives <= 0.0:
        return 1.0
    return max(1.0, negatives / positives)


@torch.no_grad()
def predict_scores(
    model: ImageResidualCalibrator,
    features: torch.Tensor,
    frozen_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], dict[str, float]]:
    model.eval()
    loader = DataLoader(TensorDataset(features, frozen_logits), batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    gate_values: list[torch.Tensor] = []
    residual_values: list[torch.Tensor] = []
    for batch_x, batch_frozen_logits in loader:
        batch_x = batch_x.to(device)
        batch_frozen_logits = batch_frozen_logits.to(device)
        outputs = model(batch_x, batch_frozen_logits, return_dict=True)
        scores.extend(torch.sigmoid(outputs["final_logits"]).cpu().tolist())
        gate_values.append(outputs["gate"].detach().cpu())
        residual_values.append(outputs["residual"].detach().cpu())
    if gate_values:
        all_gates = torch.cat(gate_values, dim=0)
        all_residuals = torch.cat(residual_values, dim=0)
        diagnostics = {
            "gate_mean": float(all_gates.mean().item()),
            "gate_max": float(all_gates.max().item()),
            "residual_abs_mean": float(all_residuals.abs().mean().item()),
            "residual_abs_max": float(all_residuals.abs().max().item()),
        }
    else:
        diagnostics = {
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "residual_abs_mean": 0.0,
            "residual_abs_max": 0.0,
        }
    return scores, diagnostics


@torch.no_grad()
def measure_identity_drift(
    model: ImageResidualCalibrator,
    features: torch.Tensor,
    frozen_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predicted, _ = predict_scores(
        model=model,
        features=features,
        frozen_logits=frozen_logits,
        batch_size=batch_size,
        device=device,
    )
    baseline = torch.sigmoid(frozen_logits).cpu().numpy().astype(np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    diff = np.abs(predicted_array - baseline)
    return {
        "max_abs_image_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_image_diff": float(diff.mean()) if diff.size else 0.0,
    }


@torch.no_grad()
def measure_rank_activity(
    model: ImageResidualCalibrator,
    features: torch.Tensor,
    frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    margin: float,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(TensorDataset(features, frozen_logits, labels), batch_size=batch_size, shuffle=False)
    rank_relu_terms: list[float] = []
    rank_smooth_terms: list[float] = []
    for batch_x, batch_frozen_logits, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_frozen_logits = batch_frozen_logits.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x, batch_frozen_logits, return_dict=True)
        rank_relu_terms.append(float(pairwise_margin_rank(outputs["final_logits"], batch_y, margin=margin, smooth=False).item()))
        rank_smooth_terms.append(float(pairwise_margin_rank(outputs["final_logits"], batch_y, margin=margin, smooth=True).item()))
    return {
        "rank_relu": float(np.mean(rank_relu_terms)) if rank_relu_terms else 0.0,
        "rank_smooth": float(np.mean(rank_smooth_terms)) if rank_smooth_terms else 0.0,
    }


def write_summary(path: Path, experiments_rows: list[dict[str, object]]) -> None:
    base = experiments_rows[0]
    cand = experiments_rows[-1]
    lines = [
        "# Stage3 P3 summary",
        "",
        (
            f"E0: image={base['image_auroc_mean']:.4f} pixel={base['pixel_auroc_mean']:.4f} "
            f"pro={base['pro_mean']:.4f} balanced={base['balanced_mean']:.4f}"
        ),
        (
            f"E1-rescal: image={cand['image_auroc_mean']:.4f} pixel={cand['pixel_auroc_mean']:.4f} "
            f"pro={cand['pro_mean']:.4f} balanced={cand['balanced_mean']:.4f}"
        ),
        (
            f"delta: image={cand['image_auroc_mean'] - base['image_auroc_mean']:+.4f} "
            f"pixel={cand['pixel_auroc_mean'] - base['pixel_auroc_mean']:+.4f} "
            f"pro={cand['pro_mean'] - base['pro_mean']:+.4f} "
            f"balanced={cand['balanced_mean'] - base['balanced_mean']:+.4f}"
        ),
        (
            f"query gate: weak5_image={cand['weak5_image_auroc_mean']:.4f} "
            f"bottle_image={cand['bottle_image_auroc']:.4f} pass={cand['gate_pass_smoke']}"
        ),
        (
            f"holdout selector: weak5_image={cand['holdout_weak5_image_auroc_mean']:.4f} "
            f"bottle_image={cand['holdout_bottle_image_auroc']:.4f}"
        ),
        f"resplits: {int(cand.get('num_resplits', 1))}",
        "note: P3 is image-only; pixel/pro stay frozen from the cached Stage2 winner during evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_residual_calibrator(
    model: ImageResidualCalibrator,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_frozen_logits: torch.Tensor,
    train_anchor_selector: torch.Tensor,
    holdout_records: list[dict[str, object]],
    holdout_x: torch.Tensor,
    holdout_frozen_logits: torch.Tensor,
    baseline_holdout_summary: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    split_index: int,
    global_step_start: int = 0,
    metrics_path: Path | None = None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float], int]:
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_frozen_logits, train_anchor_selector),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = _binary_pos_weight(train_y)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_holdout = dict(baseline_holdout_summary)
    best_holdout["selection_epoch"] = 0.0
    best_holdout["selection_metric"] = baseline_holdout_summary["weak5_image_auroc_mean"]
    best_holdout["selection_rank_proxy"] = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    global_step = global_step_start

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
            loss = (
                args.rank_weight * rank
                + args.img_bce_weight * img_bce
                + args.anchor_weight * anchor
                + args.residual_l2_weight * residual_l2
            )

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
        bottle_ok = holdout_summary["bottle_image_auroc"] >= baseline_holdout_summary["bottle_image_auroc"] - args.bottle_drop_tolerance
        selection_metric = holdout_summary["weak5_image_auroc_mean"]
        improved = (
            bottle_ok
            and (
                selection_metric > float(best_holdout["selection_metric"]) + args.min_delta
                or (
                    abs(selection_metric - float(best_holdout["selection_metric"])) <= args.min_delta
                    and (
                        holdout_rank_proxy > float(best_holdout["selection_rank_proxy"]) + args.min_delta
                        or (
                            abs(holdout_rank_proxy - float(best_holdout["selection_rank_proxy"])) <= args.min_delta
                            and holdout_summary["overall_image_auroc_mean"] > best_holdout["overall_image_auroc_mean"] + args.min_delta
                        )
                    )
                )
            )
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_holdout = {
                **holdout_summary,
                "selection_epoch": float(epoch),
                "selection_metric": selection_metric,
                "selection_rank_proxy": holdout_rank_proxy,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        row = {
            "split": float(split_index),
            "epoch": epoch,
            "global_step": float(global_step + 1),
            **{key: value / max(1, steps) for key, value in sums.items()},
            "holdout_overall_image_auroc_mean": holdout_summary["overall_image_auroc_mean"],
            "holdout_weak5_image_auroc_mean": holdout_summary["weak5_image_auroc_mean"],
            "holdout_bottle_image_auroc": holdout_summary["bottle_image_auroc"],
            "holdout_gate_mean": holdout_diag["gate_mean"],
            "holdout_residual_abs_mean": holdout_diag["residual_abs_mean"],
            "holdout_rank_proxy": holdout_rank_proxy,
            "bottle_ok": float(bottle_ok),
            "selected": float(improved),
            "selection_metric": selection_metric,
            "selection_rank_proxy": holdout_rank_proxy,
            "pos_weight": pos_weight_value,
        }
        history.append(row)
        append_jsonl(metrics_path, {"mode": MODE_P3, **row})
        global_step += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))

        if stale_epochs >= args.patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "split": split_index,
                        "epoch": epoch,
                        "best_epoch": best_holdout["selection_epoch"],
                    }
                )
            )
            break

    return history, best_state, best_holdout, global_step


def summarize_holdout_aggregate(best_holdouts: list[dict[str, float]]) -> dict[str, float]:
    return {
        "overall_image_auroc_mean": float(np.mean([row["overall_image_auroc_mean"] for row in best_holdouts])),
        "weak5_image_auroc_mean": float(np.mean([row["weak5_image_auroc_mean"] for row in best_holdouts])),
        "bottle_image_auroc": float(np.mean([row["bottle_image_auroc"] for row in best_holdouts])),
        "selection_epoch": float(np.mean([row["selection_epoch"] for row in best_holdouts])),
        "selection_metric": float(np.mean([row["selection_metric"] for row in best_holdouts])),
        "selection_rank_proxy": float(np.mean([row["selection_rank_proxy"] for row in best_holdouts])),
    }


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

    records, image_size = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    all_train_labels = build_label_tensor(train_records)
    all_train_scores = torch.tensor(
        [float(record["winner_image_score"]) for record in train_records],
        dtype=torch.float32,
    )
    baseline_image_calibration = fit_image_score_calibration(all_train_scores, all_train_labels)
    baseline_eval_scores = torch.sigmoid(build_frozen_logits(eval_records, baseline_image_calibration)).cpu().tolist()

    baseline_pixel_scores = [record["pixel_map"] for record in eval_records]
    baseline_rows = evaluate_per_category(
        eval_records=eval_records,
        image_scores=baseline_eval_scores,
        pixel_scores=baseline_pixel_scores,
        image_size=image_size,
    )
    baseline_summary = aggregate_rows(
        baseline_rows,
        experiment="E0",
        pixel_source="frozen_subspace_cache",
        balanced_mode="frozen_image_plus_frozen_pixel",
    )
    baseline_eval_image_rows = evaluate_image_per_category(eval_records, baseline_eval_scores)
    baseline_eval_image_summary = summarize_image_rows(baseline_eval_image_rows)

    per_category_rows = [{"experiment": "E0", **row} for row in baseline_rows]
    experiments_rows = [baseline_summary]
    split_histories: list[dict[str, float]] = []
    split_payloads: list[dict[str, object]] = []
    split_identity_checks: list[dict[str, float]] = []
    split_rank_checks: list[dict[str, float]] = []
    split_best_holdouts: list[dict[str, float]] = []
    holdout_per_category_rows: list[dict[str, object]] = []
    eval_score_arrays: list[np.ndarray] = []
    eval_diagnostics_rows: list[dict[str, float]] = []
    holdout_diagnostics_rows: list[dict[str, float]] = []
    global_step = 0

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
        train_x_raw, train_frozen_logits, train_hard_logits = build_feature_tensor(
            support_train_records,
            base_calibration=image_calibration,
            hard_calibration=hard_image_calibration,
        )
        holdout_x_raw, holdout_frozen_logits, holdout_hard_logits = build_feature_tensor(
            holdout_records,
            base_calibration=image_calibration,
            hard_calibration=hard_image_calibration,
        )
        eval_x_raw, eval_frozen_logits, eval_hard_logits = build_feature_tensor(
            eval_records,
            base_calibration=image_calibration,
            hard_calibration=hard_image_calibration,
        )
        train_x, holdout_x, eval_x, feature_stats = normalize_feature_splits(train_x_raw, holdout_x_raw, eval_x_raw)
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
            raise RuntimeError(f"P3 identity contract drifted before training on split {split_index + 1}: {identity_check}")
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
            raise RuntimeError(f"P3 rank objective is inactive before training on split {split_index + 1}: {rank_check}")

        history, best_state, best_holdout, global_step = train_residual_calibrator(
            model=model,
            train_x=train_x,
            train_y=train_y,
            train_frozen_logits=train_frozen_logits,
            train_anchor_selector=train_anchor_selector,
            holdout_records=holdout_records,
            holdout_x=holdout_x,
            holdout_frozen_logits=holdout_frozen_logits,
            baseline_holdout_summary=baseline_holdout_summary,
            args=args,
            device=device,
            split_index=split_index + 1,
            global_step_start=global_step,
            metrics_path=metrics_path,
        )
        model.load_state_dict(best_state)

        image_scores, eval_diag = predict_scores(
            model=model,
            features=eval_x,
            frozen_logits=eval_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )
        holdout_scores, holdout_diag = predict_scores(
            model=model,
            features=holdout_x,
            frozen_logits=holdout_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )

        eval_score_arrays.append(np.asarray(image_scores, dtype=np.float32))
        eval_diagnostics_rows.append(eval_diag)
        holdout_diagnostics_rows.append(holdout_diag)
        split_histories.extend(history)
        split_identity_checks.append(identity_check)
        split_rank_checks.append(rank_check)
        split_best_holdouts.append(best_holdout)
        holdout_per_category_rows.extend({"experiment": f"E0-s{split_index + 1}", **row} for row in baseline_holdout_rows)
        holdout_selected_rows = evaluate_image_per_category(holdout_records, holdout_scores)
        holdout_per_category_rows.extend({"experiment": f"E1-rescal-s{split_index + 1}", **row} for row in holdout_selected_rows)
        split_payloads.append(
            {
                "split": split_index + 1,
                "feature_stats": feature_stats,
                "image_score_calibration": image_calibration,
                "hard_image_score_calibration": hard_image_calibration,
                "identity_check": identity_check,
                "initial_rank_activity": rank_check,
                "baseline_holdout_summary": baseline_holdout_summary,
                "best_holdout": best_holdout,
                "holdout_split": {
                    "holdout_fraction": args.holdout_fraction,
                    "train_records": len(support_train_records),
                    "holdout_records": len(holdout_records),
                    "anchor_count": int(train_anchor_selector.sum().item()),
                    "train_by_category": {
                        category: {
                            "normal": sum(
                                1
                                for record in support_train_records
                                if str(record["category"]) == category and int(record["label"]) == 0
                            ),
                            "defect": sum(
                                1
                                for record in support_train_records
                                if str(record["category"]) == category and int(record["label"]) == 1
                            ),
                        }
                        for category in sorted({str(record["category"]) for record in train_records})
                    },
                    "holdout_by_category": {
                        category: {
                            "normal": sum(
                                1 for record in holdout_records if str(record["category"]) == category and int(record["label"]) == 0
                            ),
                            "defect": sum(
                                1 for record in holdout_records if str(record["category"]) == category and int(record["label"]) == 1
                            ),
                        }
                        for category in sorted({str(record["category"]) for record in train_records})
                    },
                },
                "train_logit_diagnostics": {
                    "base_abs_mean": float(train_frozen_logits.abs().mean().item()),
                    "hard_gap_abs_mean": float((train_hard_logits - train_frozen_logits).abs().mean().item()),
                },
                "holdout_logit_diagnostics": {
                    "base_abs_mean": float(holdout_frozen_logits.abs().mean().item()),
                    "hard_gap_abs_mean": float((holdout_hard_logits - holdout_frozen_logits).abs().mean().item()),
                },
                "eval_logit_diagnostics": {
                    "base_abs_mean": float(eval_frozen_logits.abs().mean().item()),
                    "hard_gap_abs_mean": float((eval_hard_logits - eval_frozen_logits).abs().mean().item()),
                },
            }
        )

    image_scores = np.mean(np.stack(eval_score_arrays, axis=0), axis=0).tolist()
    best_holdout = summarize_holdout_aggregate(split_best_holdouts)
    eval_diag = {
        "gate_mean": float(np.mean([row["gate_mean"] for row in eval_diagnostics_rows])),
        "gate_max": float(np.max([row["gate_max"] for row in eval_diagnostics_rows])),
        "residual_abs_mean": float(np.mean([row["residual_abs_mean"] for row in eval_diagnostics_rows])),
        "residual_abs_max": float(np.max([row["residual_abs_max"] for row in eval_diagnostics_rows])),
    }
    holdout_diag = {
        "gate_mean": float(np.mean([row["gate_mean"] for row in holdout_diagnostics_rows])),
        "gate_max": float(np.max([row["gate_max"] for row in holdout_diagnostics_rows])),
        "residual_abs_mean": float(np.mean([row["residual_abs_mean"] for row in holdout_diagnostics_rows])),
        "residual_abs_max": float(np.max([row["residual_abs_max"] for row in holdout_diagnostics_rows])),
    }
    identity_check = {
        "max_abs_image_diff": float(np.max([row["max_abs_image_diff"] for row in split_identity_checks])),
        "mean_abs_image_diff": float(np.mean([row["mean_abs_image_diff"] for row in split_identity_checks])),
    }
    rank_check = {
        "rank_relu": float(np.mean([row["rank_relu"] for row in split_rank_checks])),
        "rank_smooth": float(np.mean([row["rank_smooth"] for row in split_rank_checks])),
    }
    stats_payload = {
        "num_resplits": args.num_resplits,
        "baseline_image_score_calibration": baseline_image_calibration,
        "identity_check": identity_check,
        "initial_rank_activity": rank_check,
        "splits": split_payloads,
    }
    (output_dir / "identity_check.json").write_text(json.dumps(identity_check, indent=2), encoding="utf-8")
    (output_dir / "rank_check.json").write_text(json.dumps(rank_check, indent=2), encoding="utf-8")
    (output_dir / "rescal_stats.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    p3_rows = evaluate_per_category(
        eval_records=eval_records,
        image_scores=image_scores,
        pixel_scores=baseline_pixel_scores,
        image_size=image_size,
    )
    p3_summary = aggregate_rows(
        p3_rows,
        experiment="E1-rescal",
        pixel_source="frozen_subspace_cache",
        balanced_mode="learned_image_plus_frozen_pixel",
    )
    image_gate_rows = evaluate_image_per_category(eval_records, image_scores)
    image_gate_summary = summarize_image_rows(image_gate_rows)
    holdout_selected_rows = evaluate_image_per_category(holdout_records, holdout_scores)
    gate_pass_smoke = (
        image_gate_summary["weak5_image_auroc_mean"] >= 0.55
        and image_gate_summary["weak5_image_auroc_mean"] >= baseline_eval_image_summary["weak5_image_auroc_mean"] + 0.03
        and image_gate_summary["bottle_image_auroc"] >= baseline_eval_image_summary["bottle_image_auroc"] - args.bottle_drop_tolerance
    )
    p3_summary.update(
        {
            **image_gate_summary,
            "holdout_weak5_image_auroc_mean": best_holdout["weak5_image_auroc_mean"],
            "holdout_bottle_image_auroc": best_holdout["bottle_image_auroc"],
            "holdout_overall_image_auroc_mean": best_holdout["overall_image_auroc_mean"],
            "selection_epoch": best_holdout["selection_epoch"],
            "num_resplits": float(args.num_resplits),
            "gate_pass_smoke": bool(gate_pass_smoke),
        }
    )
    per_category_rows.extend({"experiment": "E1-rescal", **row} for row in p3_rows)
    experiments_rows.append(p3_summary)

    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "holdout_per_category.csv", holdout_per_category_rows)
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(
            {
                "baseline": "per-split baseline is recorded under splits[].baseline_holdout_summary",
                "selected": best_holdout,
                "holdout_diagnostics": holdout_diag,
                "eval_diagnostics": eval_diag,
                "num_resplits": args.num_resplits,
                "splits": [
                    {
                        "split": payload["split"],
                        "baseline_holdout_summary": payload["baseline_holdout_summary"],
                        "best_holdout": payload["best_holdout"],
                    }
                    for payload in split_payloads
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", experiments_rows)
    (output_dir / "train_history.json").write_text(json.dumps(split_histories, indent=2), encoding="utf-8")
    torch.save(
        {
            "split_payloads": split_payloads,
            "baseline_image_score_calibration": baseline_image_calibration,
            "config": vars(args),
        },
        output_dir / "image_residual_calibrator.pt",
    )

    result = {
        "mode": MODE_P3,
        "output_dir": str(output_dir),
        "baseline": experiments_rows[0],
        "candidate": experiments_rows[-1],
        "delta_image_auroc": p3_summary["image_auroc_mean"] - experiments_rows[0]["image_auroc_mean"],
        "delta_balanced": p3_summary["balanced_mean"] - experiments_rows[0]["balanced_mean"],
        "identity_check": identity_check,
        "rank_check": rank_check,
        "gate_pass_smoke": gate_pass_smoke,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
