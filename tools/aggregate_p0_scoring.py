import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)
GROUP_KEYS = (
    "stage",
    "score_mode",
    "aggregation_mode",
    "aggregation_stage",
    "topk_ratio",
    "feature_layer",
    "prototype_family",
    "num_prototypes",
    "category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cached P0 scoring summaries across seeds.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def infer_seed(input_dir: Path) -> int:
    match = SEED_PATTERN.search(input_dir.name)
    if not match:
        raise ValueError(f"Could not infer seed from directory name: {input_dir}")
    return int(match.group(1))


def load_rows(input_dir: Path) -> list[dict[str, object]]:
    seed = infer_seed(input_dir)
    metrics_paths = sorted(input_dir.glob("*/*/metrics.json"))
    if metrics_paths:
        normalized: list[dict[str, object]] = []
        for metrics_path in metrics_paths:
            config_path = metrics_path.with_name("config.json")
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing config file for metrics: {config_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            normalized.append(
                {
                    "seed": seed,
                    "stage": config["stage"],
                    "score_mode": config["score_mode"],
                    "aggregation_mode": config["aggregation_mode"],
                    "aggregation_stage": config["aggregation_stage"],
                    "topk_ratio": config["topk_ratio"],
                    "feature_layer": config["feature_layer"],
                    "prototype_family": config.get("prototype_family", "mean"),
                    "num_prototypes": config.get("num_prototypes", 1),
                    "category": metrics["category"],
                    "image_auroc": metrics["image_auroc"],
                    "pixel_auroc": metrics["pixel_auroc"],
                    "run_name": metrics_path.parents[1].name,
                    "run_output_dir": str(metrics_path.parent.resolve()),
                }
            )
        return normalized

    summary_path = input_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary file and metrics tree under: {input_dir}")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized_row = dict(row)
        normalized_row["seed"] = seed
        normalized_row.setdefault("prototype_family", "mean")
        normalized_row.setdefault("num_prototypes", 1)
        normalized.append(normalized_row)
    return normalized


def group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in GROUP_KEYS)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for key, group in grouped.items():
        image_values = [float(item["image_auroc"]) for item in group]
        pixel_values = [float(item["pixel_auroc"]) for item in group]
        seeds = sorted(int(item["seed"]) for item in group)
        aggregated_row = {name: value for name, value in zip(GROUP_KEYS, key)}
        aggregated_row.update(
            {
                "num_seeds": len(group),
                "seeds": " ".join(str(seed) for seed in seeds),
                "image_auroc_mean": mean(image_values),
                "image_auroc_std": 0.0 if len(image_values) == 1 else pstdev(image_values),
                "image_auroc_min": min(image_values),
                "image_auroc_max": max(image_values),
                "pixel_auroc_mean": mean(pixel_values),
                "pixel_auroc_std": 0.0 if len(pixel_values) == 1 else pstdev(pixel_values),
                "pixel_auroc_min": min(pixel_values),
                "pixel_auroc_max": max(pixel_values),
            }
        )
        aggregated_row["balanced_mean"] = (
            aggregated_row["image_auroc_mean"] + aggregated_row["pixel_auroc_mean"]
        ) / 2.0
        aggregated_row["balanced_floor"] = min(
            aggregated_row["image_auroc_mean"],
            aggregated_row["pixel_auroc_mean"],
        )
        aggregated.append(aggregated_row)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def top_rows(rows: list[dict[str, object]], metric: str, topn: int = 10) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item[metric]),
            float(item["balanced_floor"]),
            -float(item["image_auroc_std"]) - float(item["pixel_auroc_std"]),
        ),
        reverse=True,
    )[:topn]


def write_markdown(path: Path, by_balanced: list[dict[str, object]], by_image: list[dict[str, object]], by_pixel: list[dict[str, object]]) -> None:
    lines = [
        "# Experiment aggregate",
        "",
        "## Top balanced",
        "",
    ]
    for index, row in enumerate(by_balanced, start=1):
        lines.append(
            f"{index}. {row['stage']} | {row['score_mode']} | {row['feature_layer']} | "
            f"{row['prototype_family']} | k={row['num_prototypes']} | "
            f"{row['aggregation_mode']} | {row['aggregation_stage']} | topk={row['topk_ratio']} | "
            f"image={row['image_auroc_mean']:.4f}+-{row['image_auroc_std']:.4f} | "
            f"pixel={row['pixel_auroc_mean']:.4f}+-{row['pixel_auroc_std']:.4f}"
        )
    lines.extend(["", "## Top image", ""])
    for index, row in enumerate(by_image, start=1):
        lines.append(
            f"{index}. {row['stage']} | {row['score_mode']} | {row['feature_layer']} | "
            f"{row['prototype_family']} | k={row['num_prototypes']} | "
            f"{row['aggregation_mode']} | {row['aggregation_stage']} | topk={row['topk_ratio']} | "
            f"image={row['image_auroc_mean']:.4f}+-{row['image_auroc_std']:.4f} | "
            f"pixel={row['pixel_auroc_mean']:.4f}+-{row['pixel_auroc_std']:.4f}"
        )
    lines.extend(["", "## Top pixel", ""])
    for index, row in enumerate(by_pixel, start=1):
        lines.append(
            f"{index}. {row['stage']} | {row['score_mode']} | {row['feature_layer']} | "
            f"{row['prototype_family']} | k={row['num_prototypes']} | "
            f"{row['aggregation_mode']} | {row['aggregation_stage']} | topk={row['topk_ratio']} | "
            f"image={row['image_auroc_mean']:.4f}+-{row['image_auroc_std']:.4f} | "
            f"pixel={row['pixel_auroc_mean']:.4f}+-{row['pixel_auroc_std']:.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = [Path(value) for value in args.input_dirs]
    raw_rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        raw_rows.extend(load_rows(input_dir))

    aggregated_rows = group_rows(raw_rows)
    output_dir = Path(args.output_dir)
    by_balanced = top_rows(aggregated_rows, metric="balanced_mean")
    by_image = top_rows(aggregated_rows, metric="image_auroc_mean")
    by_pixel = top_rows(aggregated_rows, metric="pixel_auroc_mean")

    write_json(output_dir / "all_runs.json", raw_rows)
    write_csv(output_dir / "all_runs.csv", raw_rows)
    write_json(output_dir / "aggregate.json", aggregated_rows)
    write_csv(output_dir / "aggregate.csv", aggregated_rows)
    write_csv(output_dir / "top_balanced.csv", by_balanced)
    write_csv(output_dir / "top_image.csv", by_image)
    write_csv(output_dir / "top_pixel.csv", by_pixel)
    write_markdown(output_dir / "summary.md", by_balanced, by_image, by_pixel)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_input_dirs": len(input_dirs),
                "num_raw_rows": len(raw_rows),
                "num_aggregated_rows": len(aggregated_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
