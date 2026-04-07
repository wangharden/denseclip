import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.cache import FeatureCacheSpec, collect_image_paths, load_feature_cache_batch, populate_feature_cache
from fewshot.data import (
    load_mask_array,
    load_shared_split_manifest,
    save_shared_split_manifest,
    stage_a1_split_from_manifest,
    stage_b_split_from_manifest,
)
from fewshot.feature_bank import (
    PROTOTYPE_FAMILIES,
    PROTOTYPE_FAMILY_MEAN,
    PROTOTYPE_FAMILY_MEMORY_BANK,
    PROTOTYPE_FAMILY_MULTI_PROTOTYPE,
    build_reference_bank,
    flatten_feature_map,
)
from fewshot.scoring import (
    AGGREGATION_MODES,
    AGGREGATION_STAGES,
    FEATURE_LAYERS,
    SCORE_MODE_NEG_NORMAL,
    SCORE_MODE_NORMAL_MINUS_DEFECT,
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
    parser = argparse.ArgumentParser(description="Run cached P1 prototype-family ablations for Stage A1/A2.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--category", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", default="outputs/feature_cache")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--feature-layer", default="layer4", choices=FEATURE_LAYERS)
    parser.add_argument("--score-mode", default="")
    parser.add_argument("--aggregation-mode", default="topk_mean", choices=AGGREGATION_MODES)
    parser.add_argument("--aggregation-stage", default="upsampled", choices=AGGREGATION_STAGES)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--prototype-families", nargs="+", default=list(PROTOTYPE_FAMILIES))
    parser.add_argument("--num-prototypes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--reference-topks", nargs="+", type=int, default=[1])
    parser.add_argument("--kmeans-iters", type=int, default=25)
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


def default_score_mode(stage: str) -> str:
    if stage == A1_STAGE:
        return SCORE_MODE_NEG_NORMAL
    return SCORE_MODE_NORMAL_MINUS_DEFECT


def validate_args(args: argparse.Namespace) -> None:
    if not args.score_mode:
        args.score_mode = default_score_mode(args.stage)
    if args.stage == A1_STAGE and args.score_mode not in A1_SCORE_MODES:
        raise ValueError(f"Stage A1 requires a normal-only score mode, got {args.score_mode}")
    if args.score_mode not in SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {args.score_mode}")
    for family in args.prototype_families:
        if family not in PROTOTYPE_FAMILIES:
            raise ValueError(f"Unsupported prototype_family: {family}")
    for reference_topk in args.reference_topks:
        if int(reference_topk) < 1:
            raise ValueError(f"reference_topk must be >= 1, got {reference_topk}")


def build_cache_spec(args: argparse.Namespace) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        cache_root=Path(args.cache_root) / args.category,
        image_size=args.image_size,
        pretrained=args.pretrained,
        feature_layer=args.feature_layer,
        seed=args.seed,
    )


def ensure_feature_cache(
    args: argparse.Namespace,
    all_paths: list[Path],
    device: torch.device,
) -> None:
    encoder = DenseClipVisualEncoder(
        pretrained=args.pretrained,
        input_resolution=args.image_size,
        freeze=True,
    ).to(device)
    encoder.eval()
    spec = build_cache_spec(args)
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
                "feature_layer": args.feature_layer,
                "cache_root": str(spec.cache_root),
                "written_entries": len(written),
            }
        )
    )


def load_cached_batches(
    args: argparse.Namespace,
    manifest,
) -> tuple[object, dict[str, torch.Tensor]]:
    spec = build_cache_spec(args)
    if args.stage == A1_STAGE:
        split = stage_a1_split_from_manifest(manifest)
        batches = {
            "support_normal": load_feature_cache_batch(
                image_paths=collect_image_paths(split.support_normal),
                spec=spec,
            ),
            "query": load_feature_cache_batch(
                image_paths=collect_image_paths(split.test_query),
                spec=spec,
            ),
        }
        return split, batches

    split = stage_b_split_from_manifest(manifest)
    if not split.support_defect:
        raise ValueError("Stage A2 requires defect support features in cache.")
    batches = {
        "support_normal": load_feature_cache_batch(
            image_paths=collect_image_paths(split.support_normal),
            spec=spec,
        ),
        "support_defect": load_feature_cache_batch(
            image_paths=collect_image_paths(split.support_defect),
            spec=spec,
        ),
        "query": load_feature_cache_batch(
            image_paths=collect_image_paths(split.test_query),
            spec=spec,
        ),
    }
    return split, batches


def reference_count(reference: torch.Tensor | None) -> int:
    if reference is None or reference.numel() == 0:
        return 0
    if reference.ndim == 1:
        return 1
    return int(reference.shape[0])


def build_family_reference(
    feature_batch: torch.Tensor,
    prototype_family: str,
    num_prototypes: int,
    seed: int,
    kmeans_iters: int,
) -> torch.Tensor:
    flattened = flatten_feature_map(feature_batch)
    return build_reference_bank(
        features=flattened,
        prototype_family=prototype_family,
        num_prototypes=num_prototypes,
        seed=seed,
        num_iters=kmeans_iters,
    )


def iter_family_configs(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    configs: list[tuple[str, int, int]] = []
    for family in args.prototype_families:
        if family == PROTOTYPE_FAMILY_MULTI_PROTOTYPE:
            for num_prototypes in args.num_prototypes:
                configs.append((family, int(num_prototypes), 1))
        elif family == PROTOTYPE_FAMILY_MEMORY_BANK:
            for reference_topk in args.reference_topks:
                configs.append((family, 1, int(reference_topk)))
        else:
            configs.append((family, 1, 1))
    return configs


def build_run_name(
    args: argparse.Namespace,
    prototype_family: str,
    num_prototypes: int,
    reference_topk: int,
    num_support_normal: int,
    num_support_defect: int,
) -> str:
    parts = [
        args.stage,
        f"sn{int(num_support_normal):03d}",
        f"sd{int(num_support_defect):03d}",
        slug(args.feature_layer),
        slug(args.score_mode),
        slug(prototype_family),
    ]
    if prototype_family == PROTOTYPE_FAMILY_MULTI_PROTOTYPE:
        parts.append(f"k{num_prototypes:03d}")
    parts.append(f"refk{int(reference_topk):03d}")
    parts.extend(
        [
            slug(args.aggregation_mode),
            slug(args.aggregation_stage),
            f"topk{ratio_tag(args.topk_ratio)}",
        ]
    )
    return "_".join(parts)


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
                "prototype_family": config["prototype_family"],
                "num_prototypes": config["num_prototypes"],
                "reference_topk": config["reference_topk"],
                "num_support_normal": len(support_paths["support_normal"]),
                "num_support_defect": len(support_paths.get("support_defect", [])),
                "num_normal_references": prototype_payload["num_normal_references"],
                "num_defect_references": prototype_payload.get("num_defect_references", 0),
                "cache_mode": "feature-level-offline-prototype-ablation",
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
            str(row.get("prototype_family", "")),
            int(row.get("num_prototypes", 0)),
            int(row.get("reference_topk", 1)),
            str(row.get("score_mode", "")),
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
    split,
    cached_batches: dict[str, torch.Tensor],
    prototype_family: str,
    num_prototypes: int,
    reference_topk: int,
) -> dict[str, object]:
    num_support_normal = len(split.support_normal)
    num_support_defect = len(manifest.support_defect)
    run_name = build_run_name(
        args=args,
        prototype_family=prototype_family,
        num_prototypes=num_prototypes,
        reference_topk=reference_topk,
        num_support_normal=num_support_normal,
        num_support_defect=num_support_defect,
    )
    run_root = Path(args.output_dir) / run_name
    run_category_dir = run_root / args.category
    metrics_path = run_category_dir / "metrics.json"

    if args.skip_existing and metrics_path.is_file():
        metrics = read_metrics(metrics_path)
        return {
            "status": "skipped_existing",
            "stage": args.stage,
            "score_mode": args.score_mode,
            "aggregation_mode": args.aggregation_mode,
            "aggregation_stage": args.aggregation_stage,
            "topk_ratio": args.topk_ratio,
            "feature_layer": args.feature_layer,
            "prototype_family": prototype_family,
            "num_prototypes": num_prototypes,
            "reference_topk": reference_topk,
            "run_name": run_name,
            "run_output_dir": str(run_category_dir),
            "metrics_path": str(metrics_path),
            "command": ["offline-feature-cache"],
            **metrics,
        }

    normal_reference = build_family_reference(
        feature_batch=cached_batches["support_normal"],
        prototype_family=prototype_family,
        num_prototypes=num_prototypes,
        seed=args.seed,
        kmeans_iters=args.kmeans_iters,
    )
    defect_reference = None
    if args.stage == A2_STAGE:
        defect_reference = build_family_reference(
            feature_batch=cached_batches["support_defect"],
            prototype_family=prototype_family,
            num_prototypes=num_prototypes,
            seed=args.seed + 1,
            kmeans_iters=args.kmeans_iters,
        )

    query_features = cached_batches["query"]
    similarity_maps = compute_similarity_maps(
        feature_map=query_features,
        normal_prototype=normal_reference,
        defect_prototype=defect_reference,
        reference_topk=reference_topk,
    )
    score_map = build_score_map(similarity_maps, score_mode=args.score_mode)
    scored = score_map_outputs(
        score_map=score_map,
        image_size=args.image_size,
        topk_ratio=args.topk_ratio,
        aggregation_mode=args.aggregation_mode,
        aggregation_stage=args.aggregation_stage,
    )

    query_samples = list(split.test_query)
    image_scores = scored["image_scores"].cpu().tolist()
    pixel_scores = [scored["upsampled_map"][index, 0].cpu().numpy() for index in range(len(query_samples))]
    pixel_masks = [load_mask_array(sample.mask_path, image_size=args.image_size) for sample in query_samples]
    image_labels = [int(sample.label or 0) for sample in query_samples]

    metrics = {
        "category": args.category,
        "score_mode": args.score_mode,
        "prototype_family": prototype_family,
        "num_support_normal": len(split.support_normal),
        "num_test_query": len(query_samples),
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
        "reference_topk": reference_topk,
    }
    support_paths = {
        "support_normal": [str(sample.path) for sample in split.support_normal],
    }
    prototype_payload = {
        "normal_prototype": normal_reference.detach().cpu(),
        "normal_reference": normal_reference.detach().cpu(),
        "support_normal_paths": support_paths["support_normal"],
        "prototype_family": prototype_family,
        "num_prototypes": num_prototypes,
        "num_normal_references": reference_count(normal_reference),
        "config": {
            "category": args.category,
            "image_size": args.image_size,
            "pretrained": args.pretrained,
            "feature_layer": args.feature_layer,
            "score_mode": args.score_mode,
            "aggregation_mode": args.aggregation_mode,
            "aggregation_stage": args.aggregation_stage,
            "topk_ratio": args.topk_ratio,
            "prototype_family": prototype_family,
            "num_prototypes": num_prototypes,
            "reference_topk": reference_topk,
            "cache_root": str(Path(args.cache_root) / args.category),
            "seed": args.seed,
        },
        "split_manifest_path": str(Path(args.split_manifest)),
    }
    prototype_filename = "stage_a1_prototype.pt"
    if args.stage == A2_STAGE:
        support_paths["support_defect"] = [str(sample.path) for sample in split.support_defect]
        prototype_payload["defect_prototype"] = defect_reference.detach().cpu()
        prototype_payload["defect_reference"] = defect_reference.detach().cpu()
        prototype_payload["support_defect_paths"] = support_paths["support_defect"]
        prototype_payload["num_defect_references"] = reference_count(defect_reference)
        metrics["num_support_defect"] = len(split.support_defect)
        prototype_filename = "stage_a2_prototype.pt"

    config = {
        **vars(args),
        "prototype_family": prototype_family,
        "num_prototypes": num_prototypes,
        "reference_topk": reference_topk,
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
        "stage": args.stage,
        "score_mode": args.score_mode,
        "aggregation_mode": args.aggregation_mode,
        "aggregation_stage": args.aggregation_stage,
        "topk_ratio": args.topk_ratio,
        "feature_layer": args.feature_layer,
        "prototype_family": prototype_family,
        "num_prototypes": num_prototypes,
        "reference_topk": reference_topk,
        "run_name": run_name,
        "run_output_dir": str(run_category_dir),
        "metrics_path": str(metrics_path),
        "command": ["offline-feature-cache"],
        **metrics,
    }


def main() -> None:
    args = parse_args()
    if args.save_visuals:
        print(json.dumps({"warning": "--save-visuals is ignored in cached offline mode"}))
    validate_args(args)

    manifest = load_shared_split_manifest(args.split_manifest)
    if manifest.category != args.category:
        raise ValueError(f"Split manifest category mismatch: expected {args.category}, found {manifest.category}")
    if args.seed < 0:
        args.seed = int((manifest.metadata or {}).get("seed", 42))
    set_seed(args.seed)

    if args.stage == A1_STAGE:
        split = stage_a1_split_from_manifest(manifest)
        all_paths = collect_image_paths(manifest.support_normal + manifest.query_eval)
    else:
        split = stage_b_split_from_manifest(manifest)
        if not split.support_defect:
            raise ValueError("Stage A2 requires at least one defect support sample in the manifest.")
        all_paths = collect_image_paths(manifest.support_normal + manifest.support_defect + manifest.query_eval)

    ensure_feature_cache(
        args=args,
        all_paths=all_paths,
        device=resolve_device(args.device),
    )
    split, cached_batches = load_cached_batches(args=args, manifest=manifest)

    all_rows: list[dict[str, object]] = []
    for prototype_family, num_prototypes, reference_topk in iter_family_configs(args):
        row = run_one(
            args=args,
            manifest=manifest,
            split=split,
            cached_batches=cached_batches,
            prototype_family=prototype_family,
            num_prototypes=num_prototypes,
            reference_topk=reference_topk,
        )
        all_rows.append(row)
        print(
            json.dumps(
                {
                    "status": row["status"],
                    "stage": row["stage"],
                    "score_mode": row["score_mode"],
                    "feature_layer": row["feature_layer"],
                    "prototype_family": row["prototype_family"],
                    "num_prototypes": row["num_prototypes"],
                    "reference_topk": row["reference_topk"],
                    "image_auroc": row.get("image_auroc"),
                    "pixel_auroc": row.get("pixel_auroc"),
                }
            )
        )

    write_summary(Path(args.output_dir), all_rows)
    print(json.dumps({"output_dir": args.output_dir, "num_runs": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
