import argparse
import json
import sys
from collections import defaultdict
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
    _binary_pos_weight,
    append_jsonl,
    apply_score_calibration,
    fit_image_score_calibration,
    load_records,
    resolve_device,
    set_seed,
    split_records,
    write_csv,
)


EXPERIMENT_BASELINE = "E0"
EXPERIMENT_CANDIDATE = "E1-rescal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 P3 image-only residual calibrator.")
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
    parser.add_argument("--gate-bias", type=float, default=-4.0)
    parser.add_argument("--rank-weight", type=float, default=0.6)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--residual-l2-weight", type=float, default=0.05)
    parser.add_argument("--bottle-penalty-weight", type=float, default=1.0)
    parser.add_argument("--rank-proxy-weight", type=float, default=0.01)
    return parser.parse_args()


def build_residual_feature_vector(record: dict[str, object]) -> list[float]:
    base = list(record["feature_vector"])
    base[-1] = float(record["winner_image_score"])
    return base


def split_support_train_holdout(
    train_records: list[dict[str, object]],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in train_records:
        grouped[(str(record["category"]), int(record["label"]))].append(record)

    rng = np.random.default_rng(seed)
    train_split: list[dict[str, object]] = []
    holdout_split: list[dict[str, object]] = []
    for key in sorted(grouped):
        items = list(grouped[key])
        order = rng.permutation(len(items)).tolist()
        holdout_count = max(1, int(round(len(items) * holdout_fraction)))
        holdout_count = min(holdout_count, len(items) - 1)
        holdout_index = set(order[:holdout_count])
        for idx, record in enumerate(items):
            if idx in holdout_index:
                holdout_split.append(record)
            else:
                train_split.append(record)

    train_labels = {int(record["label"]) for record in train_split}
    holdout_labels = {int(record["label"]) for record in holdout_split}
    if train_labels != {0, 1} or holdout_labels != {0, 1}:
        raise ValueError(
            f"Support-holdout split must preserve both labels in train/holdout, got train={sorted(train_labels)} holdout={sorted(holdout_labels)}"
        )
    return train_split, holdout_split


def normalize_residual_features(
    train_records: list[dict[str, object]],
    holdout_records: list[dict[str, object]],
    eval_records: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    train_x = torch.tensor([build_residual_feature_vector(record) for record in train_records], dtype=torch.float32)
    holdout_x = torch.tensor([build_residual_feature_vector(record) for record in holdout_records], dtype=torch.float32)
    eval_x = torch.tensor([build_residual_feature_vector(record) for record in eval_records], dtype=torch.float32)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (
        (train_x - mean) / std,
        (holdout_x - mean) / std,
        (eval_x - mean) / std,
        {"mean": mean.tolist(), "std": std.tolist()},
    )


def build_label_tensor(records: list[dict[str, object]]) -> torch.Tensor:
    return torch.tensor([float(record["label"]) for record in records], dtype=torch.float32)


def build_frozen_score_tensor(records: list[dict[str, object]]) -> torch.Tensor:
    return torch.tensor([float(record["winner_image_score"]) for record in records], dtype=torch.float32)


def build_anchor_selector(records: list[dict[str, object]]) -> torch.Tensor:
    return torch.tensor(
        [int(record["label"]) == 0 or str(record["category"]) == "bottle" for record in records],
        dtype=torch.bool,
    )


def evaluate_image_rows(
    records: list[dict[str, object]],
    image_scores: list[float],
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, object], float]]] = defaultdict(list)
    for record, score in zip(records, image_scores, strict=True):
        grouped[str(record["category"])].append((record, float(score)))

    rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        labels = [int(record["label"]) for record, _ in items]
        scores = [score for _, score in items]
        rows.append({"category": category, "image_auroc": binary_auroc(labels, scores)})
    return rows


def summarize_image_rows(rows: list[dict[str, object]], experiment: str) -> dict[str, object]:
    weak_rows = [row for row in rows if row["category"] != "bottle"]
    bottle_row = next(row for row in rows if row["category"] == "bottle")
    return {
        "experiment": experiment,
        "num_categories": len(rows),
        "image_auroc_mean": float(np.mean([row["image_auroc"] for row in rows])),
        "weak_image_auroc_mean": float(np.mean([row["image_auroc"] for row in weak_rows])),
        "bottle_image_auroc": float(bottle_row["image_auroc"]),
    }


def select_score(
    holdout_summary: dict[str, object],
    bottle_baseline: float,
    rank_proxy: float,
    bottle_penalty_weight: float,
    rank_proxy_weight: float,
) -> float:
    weak_mean = float(holdout_summary["weak_image_auroc_mean"])
    bottle = float(holdout_summary["bottle_image_auroc"])
    bottle_penalty = max(0.0, bottle_baseline - bottle)
    return weak_mean - bottle_penalty_weight * bottle_penalty - rank_proxy_weight * rank_proxy


@torch.no_grad()
def predict_logits(
    model: ImageResidualCalibrator,
    features: torch.Tensor,
    frozen_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], torch.Tensor]:
    model.eval()
    loader = DataLoader(TensorDataset(features, frozen_logits), batch_size=batch_size, shuffle=False)
    score_rows: list[torch.Tensor] = []
    for batch_x, batch_frozen in loader:
        outputs = model(batch_x.to(device), batch_frozen.to(device))
        score_rows.append(outputs["final_logits"].detach().cpu())
    logits = torch.cat(score_rows, dim=0) if score_rows else torch.empty(0, dtype=torch.float32)
    return logits.tolist(), logits


@torch.no_grad()
def measure_identity(
    model: ImageResidualCalibrator,
    features: torch.Tensor,
    frozen_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    scores, logits = predict_logits(model, features, frozen_logits, batch_size=batch_size, device=device)
    frozen = frozen_logits.detach().cpu()
    diff = torch.abs(logits - frozen)
    return {
        "max_abs_logit_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_logit_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


@torch.no_grad()
def measure_initial_rank(
    frozen_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> dict[str, float]:
    return {
        "rank_relu": float(pairwise_margin_rank(frozen_logits, labels, margin=margin, smooth=False).item()),
        "rank_smooth": float(pairwise_margin_rank(frozen_logits, labels, margin=margin, smooth=True).item()),
    }


def train_calibrator(
    model: ImageResidualCalibrator,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_frozen_logits: torch.Tensor,
    train_anchor_selector: torch.Tensor,
    holdout_x: torch.Tensor,
    holdout_y: torch.Tensor,
    holdout_frozen_logits: torch.Tensor,
    holdout_records: list[dict[str, object]],
    holdout_bottle_baseline: float,
    args: argparse.Namespace,
    device: torch.device,
    metrics_path: Path,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float]]:
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_frozen_logits, train_anchor_selector),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = _binary_pos_weight(train_y)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics = {
        "selection_score": float("-inf"),
        "holdout_image_auroc_mean": 0.0,
        "holdout_weak_image_auroc_mean": 0.0,
        "holdout_bottle_image_auroc": 0.0,
        "holdout_rank_proxy": 0.0,
        "epoch": 0.0,
    }
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "rank": 0.0, "bce": 0.0, "anchor": 0.0, "residual_l2": 0.0}
        steps = 0
        for batch_x, batch_y, batch_frozen, batch_anchor_selector in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_frozen = batch_frozen.to(device)
            batch_anchor_selector = batch_anchor_selector.to(device)

            outputs = model(batch_x, batch_frozen)
            final_logits = outputs["final_logits"]
            residual = outputs["residual"]
            rank = pairwise_margin_rank(final_logits, batch_y, margin=args.margin, smooth=True)
            bce = F.binary_cross_entropy_with_logits(final_logits, batch_y, pos_weight=pos_weight)
            if bool(batch_anchor_selector.any().item()):
                anchor = F.mse_loss(final_logits[batch_anchor_selector], batch_frozen[batch_anchor_selector])
            else:
                anchor = final_logits.new_zeros(())
            residual_l2 = residual.square().mean()
            loss = (
                args.rank_weight * rank
                + args.bce_weight * bce
                + args.anchor_weight * anchor
                + args.residual_l2_weight * residual_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sums["loss"] += float(loss.item())
            sums["rank"] += float(rank.item())
            sums["bce"] += float(bce.item())
            sums["anchor"] += float(anchor.item())
            sums["residual_l2"] += float(residual_l2.item())
            steps += 1

        holdout_scores, holdout_logits = predict_logits(
            model=model,
            features=holdout_x,
            frozen_logits=holdout_frozen_logits,
            batch_size=args.batch_size,
            device=device,
        )
        holdout_rows = evaluate_image_rows(holdout_records, holdout_scores)
        holdout_summary = summarize_image_rows(holdout_rows, experiment="holdout")
        holdout_rank_proxy = float(pairwise_margin_rank(holdout_logits, holdout_y, margin=args.margin, smooth=False).item())
        selection_score = select_score(
            holdout_summary=holdout_summary,
            bottle_baseline=holdout_bottle_baseline,
            rank_proxy=holdout_rank_proxy,
            bottle_penalty_weight=args.bottle_penalty_weight,
            rank_proxy_weight=args.rank_proxy_weight,
        )

        row = {"epoch": epoch, **{key: value / max(1, steps) for key, value in sums.items()}}
        row["pos_weight"] = pos_weight_value
        row["holdout_image_auroc_mean"] = float(holdout_summary["image_auroc_mean"])
        row["holdout_weak_image_auroc_mean"] = float(holdout_summary["weak_image_auroc_mean"])
        row["holdout_bottle_image_auroc"] = float(holdout_summary["bottle_image_auroc"])
        row["holdout_rank_proxy"] = holdout_rank_proxy
        row["selection_score"] = selection_score
        history.append(row)
        append_jsonl(metrics_path, {"mode": "p3", **row})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))

        if selection_score > best_metrics["selection_score"] + args.min_delta:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "selection_score": selection_score,
                "holdout_image_auroc_mean": float(holdout_summary["image_auroc_mean"]),
                "holdout_weak_image_auroc_mean": float(holdout_summary["weak_image_auroc_mean"]),
                "holdout_bottle_image_auroc": float(holdout_summary["bottle_image_auroc"]),
                "holdout_rank_proxy": holdout_rank_proxy,
                "epoch": float(epoch),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(json.dumps({"event": "early_stop", "epoch": epoch, "best_selection_score": best_metrics["selection_score"]}))
                break

    return history, best_state, best_metrics


def write_summary(
    path: Path,
    experiments_rows: list[dict[str, object]],
    best_holdout_metrics: dict[str, float],
) -> None:
    lines = ["# Stage3 P3 summary", ""]
    for row in experiments_rows:
        lines.append(
            f"{row['experiment']}: image={row['image_auroc_mean']:.4f} "
            f"weak_image={row['weak_image_auroc_mean']:.4f} bottle_image={row['bottle_image_auroc']:.4f}"
        )
    if len(experiments_rows) >= 2:
        base = experiments_rows[0]
        cand = experiments_rows[-1]
        lines.append(
            f"delta: image={cand['image_auroc_mean'] - base['image_auroc_mean']:+.4f} "
            f"weak_image={cand['weak_image_auroc_mean'] - base['weak_image_auroc_mean']:+.4f} "
            f"bottle_image={cand['bottle_image_auroc'] - base['bottle_image_auroc']:+.4f}"
        )
    lines.append(
        "best_holdout: "
        f"epoch={int(best_holdout_metrics['epoch'])} "
        f"weak_image={best_holdout_metrics['holdout_weak_image_auroc_mean']:.4f} "
        f"bottle_image={best_holdout_metrics['holdout_bottle_image_auroc']:.4f} "
        f"rank_proxy={best_holdout_metrics['holdout_rank_proxy']:.4f} "
        f"selection_score={best_holdout_metrics['selection_score']:.4f}"
    )
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
    support_records, eval_records = split_records(records)
    train_records, holdout_records = split_support_train_holdout(
        support_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    train_x, holdout_x, eval_x, feature_stats = normalize_residual_features(train_records, holdout_records, eval_records)
    train_y = build_label_tensor(train_records)
    holdout_y = build_label_tensor(holdout_records)
    train_anchor_selector = build_anchor_selector(train_records)

    train_frozen_scores = build_frozen_score_tensor(train_records)
    holdout_frozen_scores = build_frozen_score_tensor(holdout_records)
    eval_frozen_scores = build_frozen_score_tensor(eval_records)
    image_calibration = fit_image_score_calibration(train_frozen_scores, train_y)
    train_frozen_logits = apply_score_calibration(train_frozen_scores, image_calibration)
    holdout_frozen_logits = apply_score_calibration(holdout_frozen_scores, image_calibration)
    eval_frozen_logits = apply_score_calibration(eval_frozen_scores, image_calibration)

    baseline_rows = evaluate_image_rows(eval_records, eval_frozen_scores.tolist())
    baseline_summary = summarize_image_rows(baseline_rows, experiment=EXPERIMENT_BASELINE)
    holdout_baseline_rows = evaluate_image_rows(holdout_records, holdout_frozen_scores.tolist())
    holdout_baseline_summary = summarize_image_rows(holdout_baseline_rows, experiment="holdout_baseline")

    model = ImageResidualCalibrator(in_features=int(train_x.shape[1]), gate_bias=args.gate_bias).to(device)
    identity_check = measure_identity(
        model=model,
        features=eval_x,
        frozen_logits=eval_frozen_logits,
        batch_size=args.batch_size,
        device=device,
    )
    initial_rank_activity = measure_initial_rank(train_frozen_logits, train_y, margin=args.margin)
    calibrator_stats = {
        "feature_stats": feature_stats,
        "image_score_calibration": image_calibration,
        "identity_check": identity_check,
        "initial_rank_activity": initial_rank_activity,
        "holdout_baseline": holdout_baseline_summary,
        "split_sizes": {
            "support_train": len(train_records),
            "support_holdout": len(holdout_records),
            "query_eval": len(eval_records),
        },
    }
    if identity_check["max_abs_logit_diff"] > 1e-6:
        raise RuntimeError(f"P3 identity contract drifted before training: {identity_check}")
    if initial_rank_activity["rank_smooth"] <= 0.0:
        raise RuntimeError(f"P3 rank objective is inactive before training: {initial_rank_activity}")
    (output_dir / "identity_check.json").write_text(json.dumps(identity_check, indent=2), encoding="utf-8")
    (output_dir / "calibrator_stats.json").write_text(json.dumps(calibrator_stats, indent=2), encoding="utf-8")

    history, best_state, best_holdout_metrics = train_calibrator(
        model=model,
        train_x=train_x,
        train_y=train_y,
        train_frozen_logits=train_frozen_logits,
        train_anchor_selector=train_anchor_selector,
        holdout_x=holdout_x,
        holdout_y=holdout_y,
        holdout_frozen_logits=holdout_frozen_logits,
        holdout_records=holdout_records,
        holdout_bottle_baseline=float(holdout_baseline_summary["bottle_image_auroc"]),
        args=args,
        device=device,
        metrics_path=metrics_path,
    )
    model.load_state_dict(best_state)
    image_scores, _ = predict_logits(
        model=model,
        features=eval_x,
        frozen_logits=eval_frozen_logits,
        batch_size=args.batch_size,
        device=device,
    )
    candidate_rows = evaluate_image_rows(eval_records, image_scores)
    candidate_summary = summarize_image_rows(candidate_rows, experiment=EXPERIMENT_CANDIDATE)

    per_category_rows = [{"experiment": EXPERIMENT_BASELINE, **row} for row in baseline_rows]
    per_category_rows.extend({"experiment": EXPERIMENT_CANDIDATE, **row} for row in candidate_rows)
    experiments_rows = [baseline_summary, candidate_summary]

    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_summary(output_dir / "summary.md", experiments_rows=experiments_rows, best_holdout_metrics=best_holdout_metrics)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "holdout_baseline.json").write_text(json.dumps(holdout_baseline_summary, indent=2), encoding="utf-8")
    (output_dir / "rank_check.json").write_text(json.dumps(initial_rank_activity, indent=2), encoding="utf-8")
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "calibrator_stats": calibrator_stats,
        "best_holdout_metrics": best_holdout_metrics,
        "config": vars(args),
    }
    torch.save(checkpoint_payload, output_dir / "image_calibrator.pt")

    result = {
        "mode": "p3_image_only",
        "output_dir": str(output_dir),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_image_auroc": candidate_summary["image_auroc_mean"] - baseline_summary["image_auroc_mean"],
        "delta_weak_image_auroc": candidate_summary["weak_image_auroc_mean"] - baseline_summary["weak_image_auroc_mean"],
        "delta_bottle_image_auroc": candidate_summary["bottle_image_auroc"] - baseline_summary["bottle_image_auroc"],
        "best_holdout_metrics": best_holdout_metrics,
        "identity_check": identity_check,
        "rank_check": initial_rank_activity,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
