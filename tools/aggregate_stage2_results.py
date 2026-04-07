import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage2 P4 full-15 results across categories and seeds.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-categories", nargs="+", default=list(MVTEC_CATEGORIES))
    return parser.parse_args()


def normalize_value(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def infer_seed(config: dict[str, object], metrics_path: Path) -> int | None:
    config_seed = config.get("seed")
    if isinstance(config_seed, int):
        return config_seed
    for candidate in metrics_path.parents:
        match = SEED_PATTERN.search(candidate.name)
        if match:
            return int(match.group(1))
    return None


def balanced_score(image_auroc: float, pixel_auroc: float) -> float:
    return (float(image_auroc) + float(pixel_auroc)) / 2.0


def scan_rows(input_dirs: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        for metrics_path in sorted(input_dir.glob("*/*/metrics.json")):
            config_path = metrics_path.with_name("config.json")
            if not config_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            image_auroc = float(metrics["image_auroc"])
            pixel_auroc = float(metrics["pixel_auroc"])
            row = {
                "input_scope": str(input_dir),
                "scope_name": input_dir.name,
                "seed": infer_seed(config, metrics_path),
                "experiment": metrics_path.parents[1].name,
                "run_output_dir": str(metrics_path.parent.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "category": str(metrics["category"]),
                "method": str(config.get("method", metrics_path.parents[1].name)),
                "feature_layer": config.get("feature_layer"),
                "score_mode": config.get("score_mode"),
                "aggregation_mode": config.get("aggregation_mode"),
                "aggregation_stage": config.get("aggregation_stage"),
                "topk_ratio": config.get("topk_ratio"),
                "reference_topk": config.get("reference_topk"),
                "subspace_dim": normalize_value(config.get("subspace_dim")),
                "num_support_normal": metrics.get("num_support_normal"),
                "num_support_defect": metrics.get("num_support_defect"),
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "pro": float(metrics["pro"]),
                "balanced_mean": balanced_score(image_auroc, pixel_auroc),
            }
            rows.append(row)
    return rows


def grouped(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        buckets.setdefault(key, []).append(row)
    return buckets


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


def aggregate_per_category(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "experiment",
        "method",
        "feature_layer",
        "score_mode",
        "aggregation_mode",
        "aggregation_stage",
        "topk_ratio",
        "reference_topk",
        "subspace_dim",
        "num_support_normal",
        "num_support_defect",
        "category",
    )
    output: list[dict[str, object]] = []
    for key, bucket in grouped(rows, keys).items():
        image_values = [float(item["image_auroc"]) for item in bucket]
        pixel_values = [float(item["pixel_auroc"]) for item in bucket]
        pro_values = [float(item["pro"]) for item in bucket]
        balanced_values = [float(item["balanced_mean"]) for item in bucket]
        seeds = sorted({item["seed"] for item in bucket if item["seed"] is not None})
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "num_rows": len(bucket),
                "num_seeds": len(seeds),
                "seeds": " ".join(str(seed) for seed in seeds),
                "image_auroc_mean": mean(image_values),
                "image_auroc_std": 0.0 if len(image_values) == 1 else pstdev(image_values),
                "pixel_auroc_mean": mean(pixel_values),
                "pixel_auroc_std": 0.0 if len(pixel_values) == 1 else pstdev(pixel_values),
                "pro_mean": mean(pro_values),
                "pro_std": 0.0 if len(pro_values) == 1 else pstdev(pro_values),
                "balanced_mean": mean(balanced_values),
                "balanced_std": 0.0 if len(balanced_values) == 1 else pstdev(balanced_values),
            }
        )
        output.append(row)
    return output


def aggregate_experiments(per_category_rows: list[dict[str, object]], expected_categories: list[str]) -> list[dict[str, object]]:
    keys = (
        "experiment",
        "method",
        "feature_layer",
        "score_mode",
        "aggregation_mode",
        "aggregation_stage",
        "topk_ratio",
        "reference_topk",
        "subspace_dim",
        "num_support_normal",
        "num_support_defect",
    )
    expected = set(expected_categories)
    output: list[dict[str, object]] = []
    for key, bucket in grouped(per_category_rows, keys).items():
        image_values = [float(item["image_auroc_mean"]) for item in bucket]
        pixel_values = [float(item["pixel_auroc_mean"]) for item in bucket]
        pro_values = [float(item["pro_mean"]) for item in bucket]
        balanced_values = [float(item["balanced_mean"]) for item in bucket]
        categories = sorted(str(item["category"]) for item in bucket)
        missing_categories = sorted(expected - set(categories))
        seed_counts = [int(item["num_seeds"]) for item in bucket]
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "num_categories": len(categories),
                "categories": " ".join(categories),
                "missing_categories": " ".join(missing_categories),
                "num_missing_categories": len(missing_categories),
                "min_seed_count": min(seed_counts) if seed_counts else 0,
                "max_seed_count": max(seed_counts) if seed_counts else 0,
                "image_auroc_mean": mean(image_values),
                "image_auroc_std": 0.0 if len(image_values) == 1 else pstdev(image_values),
                "pixel_auroc_mean": mean(pixel_values),
                "pixel_auroc_std": 0.0 if len(pixel_values) == 1 else pstdev(pixel_values),
                "pro_mean": mean(pro_values),
                "pro_std": 0.0 if len(pro_values) == 1 else pstdev(pro_values),
                "balanced_mean": mean(balanced_values),
                "balanced_std": 0.0 if len(balanced_values) == 1 else pstdev(balanced_values),
            }
        )
        output.append(row)
    output.sort(key=lambda row: (float(row["balanced_mean"]), float(row["image_auroc_mean"])), reverse=True)
    return output


def category_ranking(per_category_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(per_category_rows)
    rows.sort(key=lambda row: (float(row["balanced_mean"]), float(row["image_auroc_mean"])), reverse=True)
    return rows


def write_markdown(path: Path, experiment_rows: list[dict[str, object]], per_category_rows: list[dict[str, object]]) -> None:
    lines = ["# Stage2 P4 aggregate", ""]
    if not experiment_rows:
        lines.append("No metrics were found.")
    for index, row in enumerate(experiment_rows, start=1):
        lines.append(
            f"{index}. {row['experiment']} | image={row['image_auroc_mean']:.4f}+-{row['image_auroc_std']:.4f} | "
            f"pixel={row['pixel_auroc_mean']:.4f}+-{row['pixel_auroc_std']:.4f} | "
            f"pro={row['pro_mean']:.4f}+-{row['pro_std']:.4f} | "
            f"balanced={row['balanced_mean']:.4f}+-{row['balanced_std']:.4f} | "
            f"missing={row['num_missing_categories']}"
        )
        if row["missing_categories"]:
            lines.append(f"   missing categories: {row['missing_categories']}")
        category_rows = [item for item in per_category_rows if item["experiment"] == row["experiment"]]
        category_rows.sort(key=lambda item: float(item["balanced_mean"]), reverse=True)
        if category_rows:
            top_row = category_rows[0]
            bottom_row = category_rows[-1]
            lines.append(
                f"   best category: {top_row['category']} ({top_row['balanced_mean']:.4f}), "
                f"worst category: {bottom_row['category']} ({bottom_row['balanced_mean']:.4f})"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = [Path(value) for value in args.input_dirs]
    raw_rows = scan_rows(input_dirs)
    per_category_rows = aggregate_per_category(raw_rows)
    experiment_rows = aggregate_experiments(per_category_rows, expected_categories=list(args.expected_categories))
    ranked_categories = category_ranking(per_category_rows)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "all_runs.json", raw_rows)
    write_csv(output_dir / "all_runs.csv", raw_rows)
    write_json(output_dir / "per_category.json", per_category_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_json(output_dir / "experiments.json", experiment_rows)
    write_csv(output_dir / "experiments.csv", experiment_rows)
    write_csv(output_dir / "category_ranking.csv", ranked_categories)
    write_markdown(output_dir / "summary.md", experiment_rows, per_category_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_raw_rows": len(raw_rows),
                "num_per_category_rows": len(per_category_rows),
                "num_experiments": len(experiment_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
