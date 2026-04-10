import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.cache import FeatureCacheSpec, collect_image_paths, load_feature_cache_batch, populate_feature_cache
from fewshot.data import load_mask_array, load_shared_split_manifest, save_shared_split_manifest, stage_a1_split_from_manifest
from fewshot.fastref import fastref_lite_normal_map
from fewshot.feature_bank import PROTOTYPE_FAMILY_MEMORY_BANK, build_reference_bank, flatten_feature_map
from fewshot.patchcore_subspace import coreset_subspace_score_map
from fewshot.scoring import (
    AGGREGATION_MODE_MAX,
    AGGREGATION_MODES,
    AGGREGATION_STAGE_PATCH,
    AGGREGATION_STAGES,
    FEATURE_LAYER_LAYER4,
    FEATURE_LAYERS,
    SCORE_MODE_NEG_NORMAL,
    reference_similarity_map,
    score_map_outputs,
)
from fewshot.stage_a1 import binary_auroc, pixel_auroc
from fewshot.retrieved_subspace import retrieved_subspace_score_map
from fewshot.subspace import SUBSPACE_MODE_LOCAL, subspace_score_map


METHOD_BASELINE = "baseline"
METHOD_FASTREF = "fastref_lite"
METHOD_MATCHING = "matching"
METHOD_SUBSPACE = "subspace"
METHOD_RETRIEVED_SUBSPACE = "retrieved_subspace"
METHOD_CORESET_SUBSPACE = "coreset_subspace"
METHODS = (
    METHOD_BASELINE,
    METHOD_FASTREF,
    METHOD_MATCHING,
    METHOD_SUBSPACE,
    METHOD_RETRIEVED_SUBSPACE,
    METHOD_CORESET_SUBSPACE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached P2 structure ablations on top of the fixed A1 memory-bank baseline.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", default="outputs/feature_cache")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--feature-layer", default=FEATURE_LAYER_LAYER4, choices=FEATURE_LAYERS)
    parser.add_argument("--score-mode", default=SCORE_MODE_NEG_NORMAL, choices=(SCORE_MODE_NEG_NORMAL,))
    parser.add_argument("--aggregation-mode", default=AGGREGATION_MODE_MAX, choices=AGGREGATION_MODES)
    parser.add_argument("--aggregation-stage", default=AGGREGATION_STAGE_PATCH, choices=AGGREGATION_STAGES)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--reference-topk", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--fastref-select-ratios", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--fastref-blend-alphas", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--fastref-steps", nargs="+", type=int, default=[1])
    parser.add_argument("--coreset-ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--match-ks", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--spatial-windows", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--subspace-dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
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


def load_cached_batches(args: argparse.Namespace, manifest):
    spec = build_cache_spec(args)
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


def iter_method_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for method in args.methods:
        if method == METHOD_BASELINE:
            configs.append({"method": method})
        elif method == METHOD_FASTREF:
            for select_ratio in args.fastref_select_ratios:
                for blend_alpha in args.fastref_blend_alphas:
                    for refine_steps in args.fastref_steps:
                        configs.append(
                            {
                                "method": method,
                                "fastref_select_ratio": float(select_ratio),
                                "fastref_blend_alpha": float(blend_alpha),
                                "fastref_steps": int(refine_steps),
                            }
                        )
        elif method == METHOD_MATCHING:
            for match_k in args.match_ks:
                for spatial_window in args.spatial_windows:
                    configs.append(
                        {
                            "method": method,
                            "match_k": int(match_k),
                            "spatial_window": int(spatial_window),
                        }
                    )
        elif method == METHOD_SUBSPACE:
            for subspace_dim in args.subspace_dims:
                configs.append(
                    {
                        "method": method,
                        "subspace_dim": int(subspace_dim),
                    }
                )
        elif method == METHOD_RETRIEVED_SUBSPACE:
            for match_k in args.match_ks:
                for spatial_window in args.spatial_windows:
                    for subspace_dim in args.subspace_dims:
                        configs.append(
                            {
                                "method": method,
                                "match_k": int(match_k),
                                "spatial_window": int(spatial_window),
                                "subspace_dim": int(subspace_dim),
                            }
                        )
        elif method == METHOD_CORESET_SUBSPACE:
            for coreset_ratio in args.coreset_ratios:
                for subspace_dim in args.subspace_dims:
                    configs.append(
                        {
                            "method": method,
                            "coreset_ratio": float(coreset_ratio),
                            "subspace_dim": int(subspace_dim),
                        }
                    )
    return configs


def build_run_name(args: argparse.Namespace, manifest, split, config: dict[str, object]) -> str:
    parts = [
        "a1",
        f"sn{len(split.support_normal):03d}",
        f"sd{len(manifest.support_defect):03d}",
        slug(args.feature_layer),
        slug(args.score_mode),
        slug(str(config["method"])),
        f"refk{int(args.reference_topk):03d}",
        slug(args.aggregation_mode),
        slug(args.aggregation_stage),
        f"topk{ratio_tag(args.topk_ratio)}",
    ]
    if config["method"] == METHOD_FASTREF:
        parts.extend(
            [
                f"sel{ratio_tag(float(config['fastref_select_ratio']))}",
                f"blend{ratio_tag(float(config['fastref_blend_alpha']))}",
                f"step{int(config['fastref_steps']):03d}",
            ]
        )
    elif config["method"] == METHOD_MATCHING:
        parts.extend(
            [
                f"mk{int(config['match_k']):03d}",
                f"win{int(config['spatial_window']):03d}",
            ]
        )
    elif config["method"] == METHOD_SUBSPACE:
        parts.append(f"dim{int(config['subspace_dim']):03d}")
    elif config["method"] == METHOD_RETRIEVED_SUBSPACE:
        parts.extend(
            [
                f"mk{int(config['match_k']):03d}",
                f"win{int(config['spatial_window']):03d}",
                f"dim{int(config['subspace_dim']):03d}",
            ]
        )
    elif config["method"] == METHOD_CORESET_SUBSPACE:
        parts.extend(
            [
                f"core{ratio_tag(float(config['coreset_ratio']))}",
                f"dim{int(config['subspace_dim']):03d}",
            ]
        )
    return "_".join(parts)


def read_metrics(metrics_path: Path) -> dict[str, object]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def save_predictions(prediction_path: Path, query_samples, image_scores: list[float]) -> None:
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
            str(row.get("method", "")),
            int(row.get("subspace_dim", 0)),
            int(row.get("match_k", 0)),
            int(row.get("spatial_window", 0)),
            float(row.get("fastref_select_ratio", 0.0)),
            float(row.get("fastref_blend_alpha", 0.0)),
            int(row.get("fastref_steps", 0)),
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


def connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    labeled, num = ndimage.label(mask > 0.5)
    return [(labeled == label_id) for label_id in range(1, num + 1)]


def trapezoid(x: np.ndarray, y: np.ndarray, x_max: float | None = None) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[finite_mask]
    y = y[finite_mask]
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            ins = int(np.searchsorted(x, x_max))
            if 0 < ins < len(x):
                y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
                correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def pro_score(
    mask_arrays: list[np.ndarray],
    score_maps: list[np.ndarray],
    max_fpr: float = 0.3,
) -> float:
    if not mask_arrays or not score_maps:
        return float("nan")

    structure = np.ones((3, 3), dtype=int)
    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(score_maps), score_maps[0].shape[0], score_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(mask_arrays):
        labeled, n_components = ndimage.label(gt_map > 0.5, structure)
        num_gt_regions += n_components

        ok_mask = labeled == 0
        num_ok_pixels += int(np.sum(ok_mask))

        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = int(np.sum(region_mask))
            if region_size > 0:
                pro_change[region_mask] = 1.0 / float(region_size)

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    if num_ok_pixels <= 0 or num_gt_regions <= 0:
        return float("nan")

    anomaly_scores_flat = np.array(score_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)

    zero = np.array([0.0])
    one = np.array([1.0])
    fprs = np.concatenate((zero, fprs, one))
    pros = np.concatenate((zero, pros, one))

    return float(trapezoid(fprs, pros, x_max=max_fpr) / max_fpr)


def baseline_anomaly_map(query_features: torch.Tensor, normal_bank: torch.Tensor, reference_topk: int) -> torch.Tensor:
    normal_map = reference_similarity_map(
        query_features,
        normal_bank,
        reference_topk=reference_topk,
    )
    return -normal_map


def fastref_anomaly_map(
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    reference_topk: int,
    config: dict[str, object],
) -> torch.Tensor:
    normal_map = fastref_lite_normal_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_features,
        reference_topk=reference_topk,
        refine_ratio=float(config["fastref_select_ratio"]),
        blend_alpha=float(config["fastref_blend_alpha"]),
        refine_steps=int(config["fastref_steps"]),
    )
    return -normal_map


def matching_anomaly_map(
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    from fewshot.matching import coordinate_matching_similarity_map

    normal_map = coordinate_matching_similarity_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_features,
        match_k=int(config["match_k"]),
        spatial_window=int(config["spatial_window"]),
    )
    return -normal_map


def subspace_anomaly_map(
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    return subspace_score_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_features,
        subspace_dim=int(config["subspace_dim"]),
        mode=str(config.get("subspace_mode", SUBSPACE_MODE_LOCAL)),
    )


def retrieved_subspace_anomaly_map(
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    return retrieved_subspace_score_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_features,
        subspace_dim=int(config["subspace_dim"]),
        retrieval_topk=int(config["match_k"]),
        spatial_window=int(config["spatial_window"]),
    )


def coreset_subspace_anomaly_map(
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    config: dict[str, object],
    seed: int,
) -> torch.Tensor:
    return coreset_subspace_score_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_features,
        subspace_dim=int(config["subspace_dim"]),
        coreset_ratio=float(config["coreset_ratio"]),
        seed=int(seed),
    )


def run_method(
    args: argparse.Namespace,
    split,
    query_features: torch.Tensor,
    support_feature_batch: torch.Tensor,
    normal_bank: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    method = str(config["method"])
    if method == METHOD_BASELINE:
        return baseline_anomaly_map(query_features, normal_bank, args.reference_topk)
    if method == METHOD_FASTREF:
        return fastref_anomaly_map(query_features, support_feature_batch, args.reference_topk, config)
    if method == METHOD_MATCHING:
        return matching_anomaly_map(query_features, support_feature_batch, config)
    if method == METHOD_SUBSPACE:
        return subspace_anomaly_map(query_features, support_feature_batch, config)
    if method == METHOD_RETRIEVED_SUBSPACE:
        return retrieved_subspace_anomaly_map(query_features, support_feature_batch, config)
    if method == METHOD_CORESET_SUBSPACE:
        return coreset_subspace_anomaly_map(query_features, support_feature_batch, config, seed=args.seed)
    raise ValueError(f"Unsupported method: {method}")


def run_one(
    args: argparse.Namespace,
    manifest,
    split,
    cached_batches: dict[str, torch.Tensor],
    config: dict[str, object],
) -> dict[str, object]:
    run_name = build_run_name(args=args, manifest=manifest, split=split, config=config)
    run_root = Path(args.output_dir) / run_name
    run_category_dir = run_root / args.category
    metrics_path = run_category_dir / "metrics.json"

    if args.skip_existing and metrics_path.is_file():
        metrics = read_metrics(metrics_path)
        return {
            "status": "skipped_existing",
            "run_name": run_name,
            "run_output_dir": str(run_category_dir),
            "metrics_path": str(metrics_path),
            "command": ["offline-feature-cache"],
            "method": config["method"],
            **config,
            **metrics,
        }

    support_feature_batch = cached_batches["support_normal"]
    query_features = cached_batches["query"]
    normal_bank = build_reference_bank(
        features=flatten_feature_map(support_feature_batch),
        prototype_family=PROTOTYPE_FAMILY_MEMORY_BANK,
        num_prototypes=1,
        seed=args.seed,
        num_iters=1,
    )

    anomaly_map = run_method(
        args=args,
        split=split,
        query_features=query_features,
        support_feature_batch=support_feature_batch,
        normal_bank=normal_bank,
        config=config,
    )
    scored = score_map_outputs(
        score_map=anomaly_map,
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
        "method": config["method"],
        "num_support_normal": len(split.support_normal),
        "num_support_defect": len(manifest.support_defect),
        "num_test_query": len(query_samples),
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
        "pro": pro_score(pixel_masks, pixel_scores),
        "reference_topk": args.reference_topk,
    }

    support_paths = {
        "support_normal": [str(sample.path) for sample in split.support_normal],
    }
    run_category_dir.mkdir(parents=True, exist_ok=True)
    save_shared_split_manifest(manifest, run_category_dir / "split_manifest.json")
    (run_category_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_category_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                **config,
                "output_dir": str(run_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_category_dir / "support_paths.json").write_text(json.dumps(support_paths, indent=2), encoding="utf-8")
    (run_category_dir / "structure_summary.json").write_text(
        json.dumps(
            {
                "method": config["method"],
                "reference_topk": args.reference_topk,
                "num_support_normal": len(split.support_normal),
                "num_support_defect": len(manifest.support_defect),
                **config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_predictions(run_category_dir / "predictions.csv", query_samples, image_scores)
    return {
        "status": "completed",
        "run_name": run_name,
        "run_output_dir": str(run_category_dir),
        "metrics_path": str(metrics_path),
        "command": ["offline-feature-cache"],
        "method": config["method"],
        **config,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    manifest = load_shared_split_manifest(args.split_manifest)
    if manifest.category != args.category:
        raise ValueError(f"Split manifest category mismatch: expected {args.category}, found {manifest.category}")
    if args.seed < 0:
        args.seed = int((manifest.metadata or {}).get("seed", 42))
    set_seed(args.seed)

    split = stage_a1_split_from_manifest(manifest)
    all_paths = collect_image_paths(manifest.support_normal + manifest.query_eval)
    ensure_feature_cache(args=args, all_paths=all_paths, device=resolve_device(args.device))
    split, cached_batches = load_cached_batches(args=args, manifest=manifest)

    all_rows: list[dict[str, object]] = []
    for config in iter_method_configs(args):
        row = run_one(
            args=args,
            manifest=manifest,
            split=split,
            cached_batches=cached_batches,
            config=config,
        )
        all_rows.append(row)
        print(
            json.dumps(
                {
                    "status": row["status"],
                    "method": row["method"],
                    "image_auroc": row.get("image_auroc"),
                    "pixel_auroc": row.get("pixel_auroc"),
                    "pro": row.get("pro"),
                }
            )
        )

    write_summary(Path(args.output_dir), all_rows)
    print(json.dumps({"output_dir": args.output_dir, "num_runs": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
