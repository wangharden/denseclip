import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)
GROUP_KEYS = (
    "method",
    "feature_layer",
    "score_mode",
    "aggregation_mode",
    "aggregation_stage",
    "topk_ratio",
    "reference_topk",
    "coreset_ratio",
    "fastref_select_ratio",
    "fastref_blend_alpha",
    "fastref_steps",
    "match_k",
    "spatial_window",
    "subspace_dim",
    "category",
    "num_support_normal",
    "num_support_defect",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cached P2 structure ablation summaries across seeds.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def infer_seed(input_dir: Path) -> int:
    match = SEED_PATTERN.search(input_dir.name)
    if not match:
        raise ValueError(f"Could not infer seed from directory name: {input_dir}")
    return int(match.group(1))


def normalize_group_value(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def load_rows(input_dir: Path) -> list[dict[str, object]]:
    seed = infer_seed(input_dir)
    metrics_paths = sorted(input_dir.glob("*/*/metrics.json"))
    rows: list[dict[str, object]] = []
    for metrics_path in metrics_paths:
        config_path = metrics_path.with_name("config.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "method": config["method"],
                "feature_layer": config["feature_layer"],
                "score_mode": config["score_mode"],
                "aggregation_mode": config["aggregation_mode"],
                "aggregation_stage": config["aggregation_stage"],
                "topk_ratio": config["topk_ratio"],
                "reference_topk": config["reference_topk"],
                "coreset_ratio": normalize_group_value(config.get("coreset_ratio")),
                "fastref_select_ratio": normalize_group_value(config.get("fastref_select_ratio")),
                "fastref_blend_alpha": normalize_group_value(config.get("fastref_blend_alpha")),
                "fastref_steps": normalize_group_value(config.get("fastref_steps")),
                "match_k": normalize_group_value(config.get("match_k")),
                "spatial_window": normalize_group_value(config.get("spatial_window")),
                "subspace_dim": normalize_group_value(config.get("subspace_dim")),
                "category": metrics["category"],
                "num_support_normal": metrics["num_support_normal"],
                "num_support_defect": metrics["num_support_defect"],
                "image_auroc": metrics["image_auroc"],
                "pixel_auroc": metrics["pixel_auroc"],
                "pro": metrics["pro"],
                "run_name": metrics_path.parents[1].name,
                "run_output_dir": str(metrics_path.parent.resolve()),
            }
        )
    return rows


def group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in GROUP_KEYS)
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, object]] = []
    for key, group in grouped.items():
        image_values = [float(item["image_auroc"]) for item in group]
        pixel_values = [float(item["pixel_auroc"]) for item in group]
        pro_values = [float(item["pro"]) for item in group]
        seeds = sorted(int(item["seed"]) for item in group)
        row = {name: value for name, value in zip(GROUP_KEYS, key)}
        row.update(
            {
                "num_seeds": len(group),
                "seeds": " ".join(str(seed) for seed in seeds),
                "image_auroc_mean": mean(image_values),
                "image_auroc_std": 0.0 if len(image_values) == 1 else pstdev(image_values),
                "pixel_auroc_mean": mean(pixel_values),
                "pixel_auroc_std": 0.0 if len(pixel_values) == 1 else pstdev(pixel_values),
                "pro_mean": mean(pro_values),
                "pro_std": 0.0 if len(pro_values) == 1 else pstdev(pro_values),
            }
        )
        row["balanced_mean"] = (row["image_auroc_mean"] + row["pixel_auroc_mean"]) / 2.0
        aggregated.append(row)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def top_rows(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row[metric]), reverse=True)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# P2 structure aggregate", ""]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. {row['method']} | image={row['image_auroc_mean']:.4f}+-{row['image_auroc_std']:.4f} | "
            f"pixel={row['pixel_auroc_mean']:.4f}+-{row['pixel_auroc_std']:.4f} | "
            f"pro={row['pro_mean']:.4f}+-{row['pro_std']:.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = [Path(value) for value in args.input_dirs]
    raw_rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        raw_rows.extend(load_rows(input_dir))
    aggregated = group_rows(raw_rows)
    output_dir = Path(args.output_dir)
    by_balanced = top_rows(aggregated, "balanced_mean")
    write_json(output_dir / "all_runs.json", raw_rows)
    write_csv(output_dir / "all_runs.csv", raw_rows)
    write_json(output_dir / "aggregate.json", aggregated)
    write_csv(output_dir / "aggregate.csv", aggregated)
    write_csv(output_dir / "top_balanced.csv", by_balanced)
    write_markdown(output_dir / "summary.md", by_balanced)
    print(json.dumps({"output_dir": str(output_dir), "num_raw_rows": len(raw_rows), "num_aggregated_rows": len(aggregated)}, indent=2))


if __name__ == "__main__":
    main()
