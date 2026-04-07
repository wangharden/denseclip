import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_stage3_head import load_records, resolve_device, set_seed, split_records, write_csv
from run_stage3_text_prior import (
    CONTROL_CATEGORY,
    apply_monotonic_update,
    build_beta_grid,
    build_tau_grid,
    evaluate_image_per_category,
    format_metric,
    split_support_holdout,
    summarize_image_rows,
)


KNN_GAP_CHANNEL_INDEX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 frozen retrieval-prior image-level probe on cached Stage2 winner outputs.")
    parser.add_argument("--cache-dir", default="outputs/stage3/cache/weak5_bottle/seed42")
    parser.add_argument("--output-dir", default="outputs/stage3/diagnostics/retrieval_prior/weak5_bottle/seed42_monotonic")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--beta-max", type=float, default=2.0)
    parser.add_argument("--beta-steps", type=int, default=21)
    parser.add_argument("--tau-quantiles", default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--bottle-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    return parser.parse_args()


def aggregate_retrieval_gap(record: dict[str, object], topk_ratio: float) -> float:
    topk_ratio = min(max(float(topk_ratio), 0.0), 1.0)
    gap_map = record["map_tensor"][KNN_GAP_CHANNEL_INDEX].reshape(-1).float()
    if gap_map.numel() == 0:
        return 0.0
    if topk_ratio <= 0.0:
        return float(gap_map.mean().item())
    topk_count = max(1, int(round(gap_map.numel() * topk_ratio)))
    topk_values = torch.topk(gap_map, k=topk_count, largest=True).values
    return float(topk_values.mean().item())


def build_retrieval_gap_by_path(records: list[dict[str, object]], topk_ratio: float) -> dict[str, float]:
    return {str(record["path"]): aggregate_retrieval_gap(record, topk_ratio=topk_ratio) for record in records}


def select_split_parameters(
    holdout_records: list[dict[str, object]],
    retrieval_gap_by_path: dict[str, float],
    beta_grid: list[float],
    tau_grid: list[float],
    bottle_drop_tolerance: float,
    min_delta: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    holdout_base_scores = [float(record["winner_image_score"]) for record in holdout_records]
    holdout_base_rows = evaluate_image_per_category(holdout_records, holdout_base_scores)
    baseline_summary = summarize_image_rows(holdout_base_rows)
    holdout_gaps = [retrieval_gap_by_path[str(record["path"])] for record in holdout_records]

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
    return best, history


def summarize_gap_stats(
    records: list[dict[str, object]],
    retrieval_gap_by_path: dict[str, float],
) -> dict[str, dict[str, float]]:
    categories = sorted({str(record["category"]) for record in records})
    summary: dict[str, dict[str, float]] = {}
    for category in categories:
        values = np.asarray(
            [retrieval_gap_by_path[str(record["path"])] for record in records if str(record["category"]) == category],
            dtype=np.float64,
        )
        summary[category] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std()) if values.size else 0.0,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    _ = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, _ = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    all_records = train_records + eval_records
    retrieval_gap_by_path = build_retrieval_gap_by_path(all_records, topk_ratio=args.topk_ratio)

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
            [retrieval_gap_by_path[str(record["path"])] for record in support_train_records],
            quantiles_arg=args.tau_quantiles,
        )
        best_split, split_history = select_split_parameters(
            holdout_records=holdout_records,
            retrieval_gap_by_path=retrieval_gap_by_path,
            beta_grid=beta_grid,
            tau_grid=tau_grid,
            bottle_drop_tolerance=args.bottle_drop_tolerance,
            min_delta=args.min_delta,
        )
        for row in split_history:
            search_history.append({"split": float(split_index + 1), **row})
        split_summaries.append(
            {
                "split": float(split_index + 1),
                "support_train_records": float(len(support_train_records)),
                "holdout_records": float(len(holdout_records)),
                **best_split,
            }
        )
        eval_gaps = [retrieval_gap_by_path[str(record["path"])] for record in eval_records]
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

    experiments_rows = [
        {
            "experiment": "E0",
            "num_categories": len(baseline_eval_rows),
            "image_auroc_mean": baseline_eval_summary["overall_image_auroc_mean"],
            "weak5_image_auroc_mean": baseline_eval_summary["weak5_image_auroc_mean"],
            "bottle_image_auroc": baseline_eval_summary["bottle_image_auroc"],
        },
        {
            "experiment": "E1-retrieval-prior",
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
        },
    ]
    per_category_rows = [{"experiment": "E0", **row} for row in baseline_eval_rows]
    per_category_rows.extend({"experiment": "E1-retrieval-prior", **row} for row in final_rows)
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    (output_dir / "holdout_summary.json").write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")
    (output_dir / "search_history.json").write_text(json.dumps(search_history, indent=2), encoding="utf-8")

    retrieval_prior_stats = {
        "num_resplits": args.num_resplits,
        "beta_grid": beta_grid,
        "tau_quantiles": [float(item.strip()) for item in args.tau_quantiles.split(",") if item.strip()],
        "topk_ratio": args.topk_ratio,
        "retrieval_gap_definition": "top-k mean of knn_gap_map, where knn_gap_map = knn_top1 - knn_top3 and k = round(H*W*topk_ratio)",
        "map_channel_index": KNN_GAP_CHANNEL_INDEX,
        "category_gap_summary": summarize_gap_stats(all_records, retrieval_gap_by_path),
        "split_summaries": split_summaries,
    }
    (output_dir / "retrieval_prior_stats.json").write_text(json.dumps(retrieval_prior_stats, indent=2), encoding="utf-8")

    summary_lines = [
        "# Stage3 retrieval-prior probe summary",
        "",
        f"E0: image={format_metric(baseline_eval_summary['overall_image_auroc_mean'])}",
        f"E1-retrieval-prior: image={format_metric(final_summary['overall_image_auroc_mean'])}",
        f"delta: image={final_summary['overall_image_auroc_mean'] - baseline_eval_summary['overall_image_auroc_mean']:+.4f}",
        (
            f"query: weak5_image={format_metric(final_summary['weak5_image_auroc_mean'])} "
            f"bottle_image={format_metric(final_summary['bottle_image_auroc'])}"
        ),
        (
            f"holdout: weak5_image={format_metric(holdout_summary['weak5_image_auroc_mean'])} "
            f"bottle_image={format_metric(holdout_summary['bottle_image_auroc'])}"
        ),
        (
            f"selection: beta_mean={holdout_summary['selection_beta_mean']:.4f} "
            f"tau_mean={holdout_summary['selection_tau_mean']:.4f} resplits={args.num_resplits}"
        ),
        "note: final_score = E0 + beta * relu(retrieval_gap - tau), using frozen image score plus cached retrieval gap only.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "mode": "retrieval_prior_probe",
                "output_dir": str(output_dir),
                "overall_image_auroc_mean": final_summary["overall_image_auroc_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
