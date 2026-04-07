import argparse
import csv
import json
import random
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.cache import FeatureCacheSpec, collect_image_paths, load_feature_cache_batch, populate_feature_cache
from fewshot.data import load_mask_array, load_shared_split_manifest, save_shared_split_manifest, stage_a1_split_from_manifest, stage_b_split_from_manifest
from fewshot.scoring import (
    AGGREGATION_MODES,
    AGGREGATION_STAGES,
    FEATURE_LAYERS,
    SCORE_MODES,
    build_score_map,
    compute_similarity_maps,
    score_map_outputs,
)
from fewshot.stage_a1 import binary_auroc, pixel_auroc


A1_STAGE = "a1"
A2_STAGE = "a2"
STAGES = (A1_STAGE, A2_STAGE)

A1_SCORE_MODES = ("one-minus-normal", "neg-normal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached P0 scorer sweeps offline for Stage A1/A2.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", default="outputs/feature_cache")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument(
        "--score-modes",
        nargs="+",
        default=[
            "one-minus-normal",
            "neg-normal",
            "defect-minus-normal",
            "normal-minus-defect",
            "blend",
        ],
    )
    parser.add_argument("--aggregation-modes", nargs="+", default=["topk_mean"])
    parser.add_argument("--aggregation-stages", nargs="+", default=["upsampled"])
    parser.add_argument("--topk-ratios", nargs="+", type=float, default=[0.1])
    parser.add_argument("--feature-layers", nargs="+", default=["local"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(value: str) -> str:
    return value.replace("-", "_").replace(".", "p")


def ratio_tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def stage_score_modes(stage: str, score_modes: list[str]) -> list[str]:
    if stage == A1_STAGE:
        return [mode for mode in score_modes if mode in A1_SCORE_MODES]
    return list(score_modes)


def build_run_name(
    stage: str,
    score_mode: str,
    aggregation_mode: str,
    aggregation_stage: str,
    topk_ratio: float,
    feature_layer: str,
) -> str:
    return (
        f"{stage}_"
        f"{slug(feature_layer)}_"
        f"{slug(score_mode)}_"
        f"{slug(aggregation_mode)}_"
        f"{slug(aggregation_stage)}_"
        f"topk{ratio_tag(topk_ratio)}"
    )


def flatten_feature_batch(feature_batch: torch.Tensor) -> torch.Tensor:
    feature_batch = F.normalize(feature_batch, dim=1)
    return feature_batch.permute(0, 2, 3, 1).reshape(-1, feature_batch.shape[1])


def mean_prototype(feature_batch: torch.Tensor) -> torch.Tensor:
    flattened = flatten_feature_batch(feature_batch)
    return F.normalize(flattened.mean(dim=0), dim=0)


def build_cache_spec(
    args: argparse.Namespace,
    feature_layer: str,
) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        cache_root=Path(args.cache_root) / args.category,
        image_size=args.image_size,
        pretrained=args.pretrained,
        feature_layer=feature_layer,
        seed=args.seed,
    )


def ensure_feature_cache(
    args: argparse.Namespace,
    feature_layers: list[str],
    all_paths: list[Path],
    device: torch.device,
) -> None:
    encoder = DenseClipVisualEncoder(
        pretrained=args.pretrained,
        input_resolution=args.image_size,
        freeze=True,
    ).to(device)
    encoder.eval()
    for feature_layer in feature_layers:
        spec = build_cache_spec(args=args, feature_layer=feature_layer)
        written = populate_feature_cache(
            encoder=encoder,
            image_paths=all_paths,
            spec=spec,
            device=device,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "event": "feature_cache",
                    "feature_layer": feature_layer,
                    "cache_root": str(spec.cache_root),
                    "written_entries": len(written),
                }
            )
        )


def load_cached_batches(
    args: argparse.Namespace,
    manifest,
    feature_layers: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    stage_a1_split = stage_a1_split_from_manifest(manifest)
    stage_a2_split = stage_b_split_from_manifest(manifest)
    query_paths = collect_image_paths(stage_a1_split.test_query)

    batches: dict[str, dict[str, torch.Tensor]] = {}
    for feature_layer in feature_layers:
        spec = build_cache_spec(args=args, feature_layer=feature_layer)
        layer_batches = {
            "support_normal": load_feature_cache_batch(
                image_paths=collect_image_paths(stage_a1_split.support_normal),
                spec=spec,
            ),
            "query": load_feature_cache_batch(
                image_paths=query_paths,
                spec=spec,
            ),
        }
        if stage_a2_split.support_defect:
            layer_batches["support_defect"] = load_feature_cache_batch(
                image_paths=collect_image_paths(stage_a2_split.support_defect),
                spec=spec,
            )
        batches[feature_layer] = layer_batches
    return batches


def read_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def save_predictions(
    prediction_path: Path,
    query_samples,
    image_scores: list[float],
) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "defect_type",
        "image_score",
        "mask_path",
        "heatmap_path",
        "overlay_path",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample, image_score in zip(query_samples, image_scores):
            writer.writerow(
                {
                    "path": str(sample.path),
                    "label": int(sample.label or 0),
                    "defect_type": sample.defect_type or "unknown",
                    "image_score": float(image_score),
                    "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
                    "heatmap_path": "",
                    "overlay_path": "",
                }
            )


def save_run_outputs(
    run_category_dir: Path,
    manifest,
    query_samples,
    image_scores: list[float],
    metrics: dict[str, object],
    config: dict[str, object],
    prototype_payload: dict[str, object],
    support_paths: dict[str, list[str]],
    prototype_filename: str,
) -> None:
    run_category_dir.mkdir(parents=True, exist_ok=True)
    save_shared_split_manifest(manifest, run_category_dir / "split_manifest.json")
    (run_category_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_category_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_category_dir / "support_paths.json").write_text(json.dumps(support_paths, indent=2), encoding="utf-8")
    prototype_path = run_category_dir / prototype_filename
    torch.save(prototype_payload, prototype_path)
    (run_category_dir / "prototype_summary.json").write_text(
        json.dumps(
            {
                "prototype_path": str(prototype_path),
                "num_support_normal": len(support_paths["support_normal"]),
                "num_support_defect": len(support_paths.get("support_defect", [])),
                "cache_mode": "feature-level-offline-scoring",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_predictions(
        prediction_path=run_category_dir / "predictions.csv",
        query_samples=query_samples,
        image_scores=image_scores,
    )


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    merged_rows: dict[str, dict[str, object]] = {}
    if json_path.is_file():
        existing_rows = json.loads(json_path.read_text(encoding="utf-8"))
        for row in existing_rows:
            run_name = str(row.get("run_name", ""))
            if run_name:
                merged_rows[run_name] = row
    for row in rows:
        merged_rows[str(row["run_name"])] = row

    ordered_rows = sorted(
        merged_rows.values(),
        key=lambda row: (
            str(row.get("stage", "")),
            str(row.get("score_mode", "")),
            str(row.get("aggregation_mode", "")),
            str(row.get("aggregation_stage", "")),
            float(row.get("topk_ratio", 0.0)),
            str(row.get("feature_layer", "")),
        ),
    )
    json_path.write_text(json.dumps(ordered_rows, indent=2), encoding="utf-8")

    fieldnames: list[str] = []
    for row in ordered_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            normalized = dict(row)
            if isinstance(normalized.get("command"), list):
                normalized["command"] = " ".join(normalized["command"])
            writer.writerow(normalized)


def run_one(
    args: argparse.Namespace,
    manifest,
    cached_batches: dict[str, dict[str, torch.Tensor]],
    stage: str,
    score_mode: str,
    aggregation_mode: str,
    aggregation_stage: str,
    topk_ratio: float,
    feature_layer: str,
) -> dict[str, object]:
    run_name = build_run_name(
        stage=stage,
        score_mode=score_mode,
        aggregation_mode=aggregation_mode,
        aggregation_stage=aggregation_stage,
        topk_ratio=topk_ratio,
        feature_layer=feature_layer,
    )
    run_root = Path(args.output_dir) / run_name
    run_category_dir = run_root / args.category
    metrics_path = run_category_dir / "metrics.json"

    if args.skip_existing and metrics_path.is_file():
        metrics = read_metrics(metrics_path)
        return {
            "status": "skipped_existing",
            "stage": stage,
            "score_mode": score_mode,
            "aggregation_mode": aggregation_mode,
            "aggregation_stage": aggregation_stage,
            "topk_ratio": topk_ratio,
            "feature_layer": feature_layer,
            "run_name": run_name,
            "run_output_dir": str(run_category_dir),
            "metrics_path": str(metrics_path),
            "command": ["offline-cache"],
            **metrics,
        }

    stage_a1_split = stage_a1_split_from_manifest(manifest)
    stage_a2_split = stage_b_split_from_manifest(manifest)
    layer_batches = cached_batches[feature_layer]
    normal_prototype = mean_prototype(layer_batches["support_normal"])
    defect_prototype = None
    if stage == A2_STAGE:
        if "support_defect" not in layer_batches:
            raise ValueError("Stage A2 requires defect support features in cache.")
        defect_prototype = mean_prototype(layer_batches["support_defect"])

    query_features = layer_batches["query"]
    similarity_maps = compute_similarity_maps(
        feature_map=query_features,
        normal_prototype=normal_prototype,
        defect_prototype=defect_prototype,
    )
    score_map = build_score_map(similarity_maps, score_mode=score_mode)
    scored = score_map_outputs(
        score_map=score_map,
        image_size=args.image_size,
        topk_ratio=topk_ratio,
        aggregation_mode=aggregation_mode,
        aggregation_stage=aggregation_stage,
    )
    query_samples = list(stage_a1_split.test_query)
    image_scores = scored["image_scores"].cpu().tolist()
    pixel_scores = [scored["upsampled_map"][index, 0].cpu().numpy() for index in range(len(query_samples))]
    pixel_masks = [load_mask_array(sample.mask_path, image_size=args.image_size) for sample in query_samples]
    image_labels = [int(sample.label or 0) for sample in query_samples]

    metrics = {
        "category": args.category,
        "score_mode": score_mode,
        "num_support_normal": len(stage_a1_split.support_normal),
        "num_test_query": len(query_samples),
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
    }
    support_paths = {
        "support_normal": [str(sample.path) for sample in stage_a1_split.support_normal],
    }
    prototype_payload = {
        "normal_prototype": normal_prototype.detach().cpu(),
        "support_normal_paths": support_paths["support_normal"],
        "config": {
            "category": args.category,
            "image_size": args.image_size,
            "pretrained": args.pretrained,
            "feature_layer": feature_layer,
            "score_mode": score_mode,
            "aggregation_mode": aggregation_mode,
            "aggregation_stage": aggregation_stage,
            "topk_ratio": topk_ratio,
            "cache_root": str(Path(args.cache_root) / args.category),
            "seed": args.seed,
        },
        "split_manifest_path": str(Path(args.split_manifest)),
    }
    prototype_filename = "stage_a1_prototype.pt"
    if stage == A2_STAGE:
        support_paths["support_defect"] = [str(sample.path) for sample in stage_a2_split.support_defect]
        prototype_payload["defect_prototype"] = defect_prototype.detach().cpu()
        prototype_payload["support_defect_paths"] = support_paths["support_defect"]
        metrics["num_support_defect"] = len(stage_a2_split.support_defect)
        prototype_filename = "stage_a2_prototype.pt"

    config = {
        **vars(args),
        "stage": stage,
        "score_mode": score_mode,
        "aggregation_mode": aggregation_mode,
        "aggregation_stage": aggregation_stage,
        "topk_ratio": topk_ratio,
        "feature_layer": feature_layer,
        "output_dir": str(run_root),
    }
    save_run_outputs(
        run_category_dir=run_category_dir,
        manifest=manifest,
        query_samples=query_samples,
        image_scores=image_scores,
        metrics=metrics,
        config=config,
        prototype_payload=prototype_payload,
        support_paths=support_paths,
        prototype_filename=prototype_filename,
    )
    return {
        "status": "completed",
        "stage": stage,
        "score_mode": score_mode,
        "aggregation_mode": aggregation_mode,
        "aggregation_stage": aggregation_stage,
        "topk_ratio": topk_ratio,
        "feature_layer": feature_layer,
        "run_name": run_name,
        "run_output_dir": str(run_category_dir),
        "metrics_path": str(metrics_path),
        "command": ["offline-cache"],
        **metrics,
    }


def main() -> None:
    args = parse_args()
    if args.save_visuals:
        print(json.dumps({"warning": "--save-visuals is ignored in cached offline mode"}))

    for feature_layer in args.feature_layers:
        if feature_layer not in FEATURE_LAYERS:
            raise ValueError(f"Unsupported feature layer: {feature_layer}")
    for aggregation_mode in args.aggregation_modes:
        if aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")
    for aggregation_stage in args.aggregation_stages:
        if aggregation_stage not in AGGREGATION_STAGES:
            raise ValueError(f"Unsupported aggregation stage: {aggregation_stage}")
    for score_mode in args.score_modes:
        if score_mode not in SCORE_MODES:
            raise ValueError(f"Unsupported score mode: {score_mode}")

    manifest = load_shared_split_manifest(args.split_manifest)
    if manifest.category != args.category:
        raise ValueError(f"Split manifest category mismatch: expected {args.category}, found {manifest.category}")
    if args.seed < 0:
        args.seed = int((manifest.metadata or {}).get("seed", 42))
    set_seed(args.seed)

    all_paths = collect_image_paths(manifest.support_normal + manifest.support_defect + manifest.query_eval)
    ensure_feature_cache(
        args=args,
        feature_layers=sorted(set(args.feature_layers)),
        all_paths=all_paths,
        device=resolve_device(args.device),
    )
    cached_batches = load_cached_batches(
        args=args,
        manifest=manifest,
        feature_layers=sorted(set(args.feature_layers)),
    )

    all_rows: list[dict[str, object]] = []
    for stage in args.stages:
        selected_score_modes = stage_score_modes(stage, args.score_modes)
        if not selected_score_modes:
            raise ValueError(f"No score modes available for stage={stage}.")
        for score_mode, aggregation_mode, aggregation_stage, topk_ratio, feature_layer in product(
            selected_score_modes,
            args.aggregation_modes,
            args.aggregation_stages,
            args.topk_ratios,
            args.feature_layers,
        ):
            row = run_one(
                args=args,
                manifest=manifest,
                cached_batches=cached_batches,
                stage=stage,
                score_mode=score_mode,
                aggregation_mode=aggregation_mode,
                aggregation_stage=aggregation_stage,
                topk_ratio=topk_ratio,
                feature_layer=feature_layer,
            )
            all_rows.append(row)
            print(
                json.dumps(
                    {
                        "status": row["status"],
                        "stage": row["stage"],
                        "score_mode": row["score_mode"],
                        "aggregation_mode": row["aggregation_mode"],
                        "aggregation_stage": row["aggregation_stage"],
                        "topk_ratio": row["topk_ratio"],
                        "feature_layer": row["feature_layer"],
                        "image_auroc": row.get("image_auroc"),
                        "pixel_auroc": row.get("pixel_auroc"),
                    }
                )
            )

    write_summary(Path(args.output_dir), all_rows)
    print(json.dumps({"output_dir": args.output_dir, "num_runs": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
