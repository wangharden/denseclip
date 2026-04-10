import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as tv_rotate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.coreset import greedy_farthest_point_coreset
from fewshot.data import (
    RESIZE_MODE_SMALLER_EDGE,
    RESIZE_MODE_SQUARE,
    RESIZE_MODES,
    build_shared_split_manifest,
    load_mask_array,
    load_shared_split_manifest,
    save_shared_split_manifest,
)
from fewshot.dinov2_backbone import (
    FEATURE_CACHE_VERSION,
    DinoV2PatchEncoder,
    FEATURE_SOURCE_LAST,
    FEATURE_SOURCE_LAST4_MEAN,
    FEATURE_SOURCES,
    FEATURE_VIEW_BASE,
    FEATURE_VIEW_OBJECT_NORMALIZED,
    flatten_patch_map,
    load_dinov2_feature_cache_batch,
    populate_dinov2_feature_cache,
)
from fewshot.scoring import AGGREGATION_MODE_TOPK_MEAN, score_map_outputs
from fewshot.stage_a1 import binary_auroc, pixel_auroc


DEFAULT_WEAK5_CATEGORIES = ("bottle", "carpet", "grid", "leather", "screw", "zipper")
DEFAULT_FULL15_CATEGORIES = (
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
METHOD_PATCHCORE_KNN = "patchcore_knn"
METHOD_PATCHCORE_DEFECT_BOOST = "patchcore_defect_boost"
METHOD_PATCHCORE_LOCAL_SCALE = "patchcore_local_scale"
METHOD_PATCHCORE_SUPPORT_CONSENSUS = "patchcore_support_consensus"
METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST = "patchcore_support_consensus_defect_boost"
METHOD_PATCHCORE_MATCH_REWEIGHT = "patchcore_match_reweight"
METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS = "patchcore_supportaware_preproc_consensus"
METHOD_GLOBAL_SUBSPACE = "global_subspace"
METHODS = (
    METHOD_PATCHCORE_KNN,
    METHOD_PATCHCORE_DEFECT_BOOST,
    METHOD_PATCHCORE_LOCAL_SCALE,
    METHOD_PATCHCORE_SUPPORT_CONSENSUS,
    METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST,
    METHOD_PATCHCORE_MATCH_REWEIGHT,
    METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS,
    METHOD_GLOBAL_SUBSPACE,
)
CONTROL_CATEGORY = "bottle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage4 pure-visual DINOv2 screening/finalist runner.")
    parser.add_argument("--mode", choices=("screening", "finalist"), default="screening")
    parser.add_argument("--subset", choices=("weak5_bottle", "full15"), required=True)
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--manifests-dir", default="outputs/split_manifests/stage4_pure_visual_dinov2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-root", default="outputs/feature_cache_dinov2/stage4_pure_visual")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--support-normal-k", type=int, default=16)
    parser.add_argument("--support-defect-k", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--resize-mode", choices=RESIZE_MODES, default=RESIZE_MODE_SQUARE)
    parser.add_argument("--resize-patch-multiple", type=int, default=14)
    parser.add_argument("--feature-sources", nargs="+", default=[FEATURE_SOURCE_LAST, FEATURE_SOURCE_LAST4_MEAN], choices=FEATURE_SOURCES)
    parser.add_argument("--methods", nargs="+", default=[METHOD_PATCHCORE_KNN, METHOD_GLOBAL_SUBSPACE], choices=METHODS)
    parser.add_argument("--knn-topks", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--defect-boost-weights", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--local-scale-neighbor-topks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--support-consensus-topks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--match-reweight-alphas", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--preproc-agreement-betas", nargs="+", type=float, default=[8.0])
    parser.add_argument("--coreset-ratios", nargs="+", type=float, default=[1.0, 0.05])
    parser.add_argument("--subspace-dims", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def default_categories(subset: str) -> tuple[str, ...]:
    if subset == "weak5_bottle":
        return DEFAULT_WEAK5_CATEGORIES
    return DEFAULT_FULL15_CATEGORIES


def default_seeds(mode: str) -> list[int]:
    if mode == "screening":
        return [42]
    return [42, 43, 44]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manifest_filename(category: str, support_normal_k: int, support_defect_k: int, seed: int) -> str:
    return f"{category}_sn{support_normal_k}_sd{support_defect_k}_seed{seed}.json"


def ensure_manifest(
    manifests_dir: Path,
    data_root: Path,
    category: str,
    support_normal_k: int,
    support_defect_k: int,
    seed: int,
) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / manifest_filename(category, support_normal_k, support_defect_k, seed)
    if manifest_path.is_file():
        manifest = load_shared_split_manifest(manifest_path)
        if manifest.category != category:
            raise ValueError(f"Manifest category mismatch at {manifest_path}: {manifest.category} != {category}")
        return manifest_path
    manifest = build_shared_split_manifest(
        root=data_root,
        category=category,
        support_normal_k=support_normal_k,
        support_defect_k=support_defect_k,
        seed=seed,
    )
    save_shared_split_manifest(manifest, manifest_path)
    return manifest_path


def slug(value: str) -> str:
    return str(value).replace("-", "_").replace(".", "p")


def ratio_tag(value: float) -> str:
    return f"{int(round(float(value) * 1000)):03d}"


def resize_mode_tag(resize_mode: str, patch_multiple: int) -> str:
    return f"{slug(resize_mode)}_pm{int(patch_multiple):03d}"


def patch_batch_output_size(patch_batch: torch.Tensor, patch_multiple: int) -> tuple[int, int]:
    if patch_batch.ndim != 4:
        raise ValueError(f"Expected a 4D patch batch, got shape {tuple(patch_batch.shape)}")
    return (
        int(patch_batch.shape[-2]) * int(patch_multiple),
        int(patch_batch.shape[-1]) * int(patch_multiple),
    )


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


def split_support_holdout(support_normal, holdout_fraction: float, seed: int):
    items = list(sorted(support_normal, key=lambda sample: str(sample.path)))
    if len(items) <= 1:
        raise ValueError("Need at least two normal support images for a holdout split.")
    rng = np.random.RandomState(seed)
    holdout_count = int(round(len(items) * float(holdout_fraction)))
    holdout_count = max(1, holdout_count)
    holdout_count = min(len(items) - 1, holdout_count)
    holdout_indices = set(rng.permutation(len(items))[:holdout_count].tolist())
    support_train = [sample for idx, sample in enumerate(items) if idx not in holdout_indices]
    support_holdout = [sample for idx, sample in enumerate(items) if idx in holdout_indices]
    return support_train, support_holdout


OFFICIAL_ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)


def _rotation_safe_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


def rotate_patch_map_and_mask(patch_map: torch.Tensor, mask: torch.Tensor, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    rotated_patch = tv_rotate(
        patch_map,
        angle=float(angle),
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0,
    )
    rotated_mask = tv_rotate(
        _rotation_safe_mask(mask).float(),
        angle=float(angle),
        interpolation=InterpolationMode.NEAREST,
        expand=False,
        fill=0,
    ).to(dtype=torch.bool)
    rotated_patch = rotated_patch * rotated_mask.to(dtype=rotated_patch.dtype)
    return rotated_patch, rotated_mask


def augment_patch_batch_with_rotations(
    patch_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    angles: tuple[int, ...] = OFFICIAL_ROTATION_ANGLES,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotated_batches: list[torch.Tensor] = []
    rotated_masks: list[torch.Tensor] = []
    for angle in angles:
        for patch_map, mask in zip(patch_batch, mask_batch, strict=True):
            rotated_patch, rotated_mask = rotate_patch_map_and_mask(patch_map, mask, angle=float(angle))
            rotated_batches.append(rotated_patch)
            rotated_masks.append(rotated_mask)
    return torch.stack(rotated_batches, dim=0), torch.stack(rotated_masks, dim=0)


def foreground_mask_from_patch_batch(patch_batch: torch.Tensor) -> torch.Tensor:
    return patch_batch.abs().sum(dim=1, keepdim=True) > 1e-6


def average_precision_score(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-scores_array)
    sorted_labels = labels_array[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def classification_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    preds = (scores_array >= float(threshold)).astype(np.int64)
    tp = int(((preds == 1) & (labels_array == 1)).sum())
    tn = int(((preds == 0) & (labels_array == 0)).sum())
    fp = int(((preds == 1) & (labels_array == 0)).sum())
    fn = int(((preds == 0) & (labels_array == 1)).sum())
    accuracy = (tp + tn) / max(1, len(labels_array))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "num_pred_positive": int(preds.sum()),
        "num_pred_negative": int((1 - preds).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(labels: list[int], scores: list[float]) -> dict[str, float]:
    values = sorted({float(score) for score in scores})
    if not values:
        return {"threshold": 0.0, "balanced_accuracy": 0.0, "f1": 0.0}
    candidates = [values[0] - 1e-6, *values, values[-1] + 1e-6]
    best = {
        "threshold": float(candidates[0]),
        "balanced_accuracy": -1.0,
        "f1": -1.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "num_pred_positive": 0,
        "num_pred_negative": 0,
    }
    for threshold in candidates:
        metrics = classification_metrics(labels=labels, scores=scores, threshold=float(threshold))
        candidate = (
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["specificity"],
            -abs(float(threshold)),
        )
        incumbent = (
            best["balanced_accuracy"],
            best["f1"],
            best["specificity"],
            -abs(float(best["threshold"])),
        )
        if candidate > incumbent:
            best = {"threshold": float(threshold), **metrics}
    return best


def choose_thresholds_by_category(records: list[dict[str, object]], scores: list[float]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "scores": []})
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        payload[category]["labels"].append(int(record["label"]))
        payload[category]["scores"].append(float(score))
    threshold_map: dict[str, dict[str, float]] = {}
    for category in sorted(payload):
        threshold_map[category] = choose_threshold(payload[category]["labels"], payload[category]["scores"])
    return threshold_map


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
            insert_at = int(np.searchsorted(x, x_max))
            if 0 < insert_at < len(x):
                y_interp = y[insert_at - 1] + ((y[insert_at] - y[insert_at - 1]) * (x_max - x[insert_at - 1]) / (x[insert_at] - x[insert_at - 1]))
                correction = 0.5 * (y_interp + y[insert_at - 1]) * (x_max - x[insert_at - 1])
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def pro_score(mask_arrays: list[np.ndarray], score_maps: list[np.ndarray], max_fpr: float = 0.3) -> float:
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


def stage4_baseline_paths(subset: str) -> dict[str, Path]:
    if subset == "weak5_bottle":
        base_dir = REPO_ROOT / "outputs" / "new-branch" / "prompt-text" / "weak5_bottle" / "seed42" / "prompt_p_v3_thresholdfix"
    else:
        base_dir = REPO_ROOT / "outputs" / "new-branch" / "prompt-text" / "full15" / "seed42" / "prompt_p_v3_full15"
    return {
        "experiments": base_dir / "experiments.csv",
        "per_category": base_dir / "per_category.csv",
        "summary": base_dir / "summary.md",
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_baseline_refs(subset: str) -> dict[str, object]:
    paths = stage4_baseline_paths(subset=subset)
    experiments = load_csv_rows(paths["experiments"])
    per_category = load_csv_rows(paths["per_category"])

    e0_row = next(row for row in experiments if row["experiment"] == "E0")
    incumbent_row = next(
        row for row in experiments if row["experiment"] == "prompt_plus_p__clip_global"
    )
    e0_per_category = {row["category"]: row for row in per_category if row["experiment"] == "E0"}
    incumbent_per_category = {row["category"]: row for row in per_category if row["experiment"] == "prompt_plus_p__clip_global"}
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "stage2_e0": e0_row,
        "stage4_incumbent": incumbent_row,
        "stage2_e0_per_category": e0_per_category,
        "stage4_incumbent_per_category": incumbent_per_category,
    }


def build_candidate_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for feature_source in args.feature_sources:
        if METHOD_PATCHCORE_KNN in args.methods:
            for knn_topk in args.knn_topks:
                for coreset_ratio in args.coreset_ratios:
                    configs.append(
                        {
                            "method": METHOD_PATCHCORE_KNN,
                            "feature_source": feature_source,
                            "knn_topk": int(knn_topk),
                            "coreset_ratio": float(coreset_ratio),
                            "topk_ratio": float(args.topk_ratio),
                            "resize_mode": str(args.resize_mode),
                            "resize_patch_multiple": int(args.resize_patch_multiple),
                        }
                    )
        if METHOD_PATCHCORE_DEFECT_BOOST in args.methods:
            for knn_topk in args.knn_topks:
                for defect_boost_weight in args.defect_boost_weights:
                    for coreset_ratio in args.coreset_ratios:
                        configs.append(
                            {
                                "method": METHOD_PATCHCORE_DEFECT_BOOST,
                                "feature_source": feature_source,
                                "knn_topk": int(knn_topk),
                                "defect_boost_weight": float(defect_boost_weight),
                                "coreset_ratio": float(coreset_ratio),
                                "topk_ratio": float(args.topk_ratio),
                                "resize_mode": str(args.resize_mode),
                                "resize_patch_multiple": int(args.resize_patch_multiple),
                            }
                        )
        if METHOD_PATCHCORE_LOCAL_SCALE in args.methods:
            for knn_topk in args.knn_topks:
                for local_scale_neighbor_topk in args.local_scale_neighbor_topks:
                    for coreset_ratio in args.coreset_ratios:
                        configs.append(
                            {
                                "method": METHOD_PATCHCORE_LOCAL_SCALE,
                                "feature_source": feature_source,
                                "knn_topk": int(knn_topk),
                                "local_scale_neighbor_topk": int(local_scale_neighbor_topk),
                                "coreset_ratio": float(coreset_ratio),
                                "topk_ratio": float(args.topk_ratio),
                                "resize_mode": str(args.resize_mode),
                                "resize_patch_multiple": int(args.resize_patch_multiple),
                            }
                        )
        if METHOD_PATCHCORE_SUPPORT_CONSENSUS in args.methods:
            for knn_topk in args.knn_topks:
                for support_consensus_topk in args.support_consensus_topks:
                    for coreset_ratio in args.coreset_ratios:
                        configs.append(
                            {
                                "method": METHOD_PATCHCORE_SUPPORT_CONSENSUS,
                                "feature_source": feature_source,
                                "knn_topk": int(knn_topk),
                                "support_consensus_topk": int(support_consensus_topk),
                                "coreset_ratio": float(coreset_ratio),
                                "topk_ratio": float(args.topk_ratio),
                                "resize_mode": str(args.resize_mode),
                                "resize_patch_multiple": int(args.resize_patch_multiple),
                            }
                        )
        if METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST in args.methods:
            for knn_topk in args.knn_topks:
                for support_consensus_topk in args.support_consensus_topks:
                    for defect_boost_weight in args.defect_boost_weights:
                        for coreset_ratio in args.coreset_ratios:
                            configs.append(
                                {
                                    "method": METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST,
                                    "feature_source": feature_source,
                                    "knn_topk": int(knn_topk),
                                    "support_consensus_topk": int(support_consensus_topk),
                                    "defect_boost_weight": float(defect_boost_weight),
                                    "coreset_ratio": float(coreset_ratio),
                                    "topk_ratio": float(args.topk_ratio),
                                    "resize_mode": str(args.resize_mode),
                                    "resize_patch_multiple": int(args.resize_patch_multiple),
                                }
                            )
        if METHOD_PATCHCORE_MATCH_REWEIGHT in args.methods:
            for knn_topk in args.knn_topks:
                for support_consensus_topk in args.support_consensus_topks:
                    for match_reweight_alpha in args.match_reweight_alphas:
                        for coreset_ratio in args.coreset_ratios:
                            configs.append(
                                {
                                    "method": METHOD_PATCHCORE_MATCH_REWEIGHT,
                                    "feature_source": feature_source,
                                    "knn_topk": int(knn_topk),
                                    "support_consensus_topk": int(support_consensus_topk),
                                    "match_reweight_alpha": float(match_reweight_alpha),
                                    "coreset_ratio": float(coreset_ratio),
                                    "topk_ratio": float(args.topk_ratio),
                                    "resize_mode": str(args.resize_mode),
                                    "resize_patch_multiple": int(args.resize_patch_multiple),
                                }
                            )
        if METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS in args.methods:
            for knn_topk in args.knn_topks:
                for support_consensus_topk in args.support_consensus_topks:
                    for preproc_agreement_beta in args.preproc_agreement_betas:
                        for coreset_ratio in args.coreset_ratios:
                            configs.append(
                                {
                                    "method": METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS,
                                    "feature_source": feature_source,
                                    "knn_topk": int(knn_topk),
                                    "support_consensus_topk": int(support_consensus_topk),
                                    "preproc_agreement_beta": float(preproc_agreement_beta),
                                    "coreset_ratio": float(coreset_ratio),
                                    "topk_ratio": float(args.topk_ratio),
                                    "resize_mode": str(args.resize_mode),
                                    "resize_patch_multiple": int(args.resize_patch_multiple),
                                }
                            )
        if METHOD_GLOBAL_SUBSPACE in args.methods:
            for subspace_dim in args.subspace_dims:
                for coreset_ratio in args.coreset_ratios:
                    configs.append(
                        {
                            "method": METHOD_GLOBAL_SUBSPACE,
                            "feature_source": feature_source,
                            "subspace_dim": int(subspace_dim),
                            "coreset_ratio": float(coreset_ratio),
                            "topk_ratio": float(args.topk_ratio),
                            "resize_mode": str(args.resize_mode),
                            "resize_patch_multiple": int(args.resize_patch_multiple),
                        }
                    )
    return configs


def candidate_name(backbone: str, config: dict[str, object]) -> str:
    method_label_map = {
        METHOD_PATCHCORE_KNN: "patchcore_knn",
        METHOD_PATCHCORE_DEFECT_BOOST: "pc_defboost",
        METHOD_PATCHCORE_LOCAL_SCALE: "pc_localscale",
        METHOD_PATCHCORE_SUPPORT_CONSENSUS: "pc_consensus",
        METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST: "pc_consensus_defboost",
        METHOD_PATCHCORE_MATCH_REWEIGHT: "pc_matchrw",
        METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS: "pc_preproc_consensus",
        METHOD_GLOBAL_SUBSPACE: "global_subspace",
    }
    method_label = method_label_map.get(str(config["method"]), str(config["method"]))
    parts = [slug(backbone), slug(str(config["feature_source"])), slug(method_label)]
    if config["method"] in (
        METHOD_PATCHCORE_KNN,
        METHOD_PATCHCORE_DEFECT_BOOST,
        METHOD_PATCHCORE_LOCAL_SCALE,
        METHOD_PATCHCORE_SUPPORT_CONSENSUS,
        METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST,
        METHOD_PATCHCORE_MATCH_REWEIGHT,
        METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS,
    ):
        parts.append(f"k{int(config['knn_topk']):03d}")
    if config["method"] == METHOD_PATCHCORE_DEFECT_BOOST:
        parts.append(f"boost{ratio_tag(float(config['defect_boost_weight']))}")
    if config["method"] == METHOD_PATCHCORE_LOCAL_SCALE:
        parts.append(f"scale{int(config['local_scale_neighbor_topk']):03d}")
    if config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS:
        parts.append(f"cons{int(config['support_consensus_topk']):03d}")
    if config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST:
        parts.append(f"cons{int(config['support_consensus_topk']):03d}")
        parts.append(f"boost{ratio_tag(float(config['defect_boost_weight']))}")
    if config["method"] == METHOD_PATCHCORE_MATCH_REWEIGHT:
        parts.append(f"cons{int(config['support_consensus_topk']):03d}")
        parts.append(f"alpha{ratio_tag(float(config['match_reweight_alpha']))}")
    if config["method"] == METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS:
        parts.append(f"cons{int(config['support_consensus_topk']):03d}")
        parts.append(f"beta{ratio_tag(float(config['preproc_agreement_beta']))}")
    if config["method"] == METHOD_GLOBAL_SUBSPACE:
        parts.append(f"dim{int(config['subspace_dim']):03d}")
    parts.append(f"core{ratio_tag(float(config['coreset_ratio']))}")
    parts.append(f"topk{ratio_tag(float(config['topk_ratio']))}")
    if (
        str(config.get("resize_mode", RESIZE_MODE_SQUARE)) != RESIZE_MODE_SQUARE
        or int(config.get("resize_patch_multiple", 14)) != 14
    ):
        parts.append(
            resize_mode_tag(
                resize_mode=str(config.get("resize_mode", RESIZE_MODE_SQUARE)),
                patch_multiple=int(config.get("resize_patch_multiple", 14)),
            )
        )
    return "_".join(parts)


def build_records(samples, role: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sample in samples:
        records.append(
            {
                "path": str(sample.path),
                "label": int(sample.label or 0),
                "category": str(sample.category),
                "defect_type": sample.defect_type or "good",
                "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
                "role": role,
            }
        )
    return records


def patchcore_knn_score_map(query_patch_map: torch.Tensor, reference_bank: torch.Tensor, knn_topk: int) -> torch.Tensor:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    reference_bank = F.normalize(reference_bank, dim=1, eps=1e-6)
    similarities = query_rows @ reference_bank.t()
    k = max(1, min(int(knn_topk), similarities.shape[1]))
    if k == 1:
        similarity = similarities.max(dim=1).values
    else:
        similarity = similarities.topk(k=k, dim=1).values.mean(dim=1)
    return (1.0 - similarity).view(batch_size, 1, height, width)


def patchcore_defect_boost_score_map(
    query_patch_map: torch.Tensor,
    normal_reference_bank: torch.Tensor,
    defect_reference_bank: torch.Tensor,
    knn_topk: int,
    defect_boost_weight: float,
) -> torch.Tensor:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    normal_reference_bank = F.normalize(normal_reference_bank, dim=1, eps=1e-6)
    defect_reference_bank = F.normalize(defect_reference_bank, dim=1, eps=1e-6)

    normal_similarities = query_rows @ normal_reference_bank.t()
    defect_similarities = query_rows @ defect_reference_bank.t()
    k_normal = max(1, min(int(knn_topk), normal_similarities.shape[1]))
    k_defect = max(1, min(int(knn_topk), defect_similarities.shape[1]))
    if k_normal == 1:
        normal_similarity = normal_similarities.max(dim=1).values
    else:
        normal_similarity = normal_similarities.topk(k=k_normal, dim=1).values.mean(dim=1)
    if k_defect == 1:
        defect_similarity = defect_similarities.max(dim=1).values
    else:
        defect_similarity = defect_similarities.topk(k=k_defect, dim=1).values.mean(dim=1)

    normal_distance = 1.0 - normal_similarity
    defect_margin = torch.relu(defect_similarity - normal_similarity)
    anomaly = normal_distance + float(defect_boost_weight) * defect_margin
    return anomaly.view(batch_size, 1, height, width)


def compute_reference_local_radii(
    reference_bank: torch.Tensor,
    neighbor_topk: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    if reference_bank.ndim != 2:
        raise ValueError(f"Expected a 2D reference bank, got {tuple(reference_bank.shape)}")
    normalized = F.normalize(reference_bank, dim=1, eps=1e-6)
    total = normalized.shape[0]
    if total <= 1:
        return normalized.new_ones((total,))
    use_k = max(1, min(int(neighbor_topk), total - 1))
    radii: list[torch.Tensor] = []
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        chunk = normalized[start:end]
        similarities = chunk @ normalized.t()
        values = similarities.topk(k=use_k + 1, dim=1).values[:, 1:]
        distances = 1.0 - values
        radii.append(distances.mean(dim=1))
    return torch.cat(radii, dim=0).contiguous()


def patchcore_local_scale_score_map(
    query_patch_map: torch.Tensor,
    reference_bank: torch.Tensor,
    reference_local_radii: torch.Tensor,
    knn_topk: int,
) -> torch.Tensor:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    reference_bank = F.normalize(reference_bank, dim=1, eps=1e-6)
    similarities = query_rows @ reference_bank.t()
    k = max(1, min(int(knn_topk), similarities.shape[1]))
    if k == 1:
        normal_similarity, nearest_index = similarities.max(dim=1)
    else:
        values, indices = similarities.topk(k=k, dim=1)
        normal_similarity = values.mean(dim=1)
        nearest_index = indices[:, 0]
    normal_distance = 1.0 - normal_similarity
    local_radius = reference_local_radii[nearest_index].clamp_min(1e-6)
    anomaly = normal_distance / local_radius
    return anomaly.view(batch_size, 1, height, width)


def patchcore_support_consensus_score_map(
    query_patch_map: torch.Tensor,
    support_patch_bank: torch.Tensor,
    knn_topk: int,
    support_consensus_topk: int,
    return_best_normal_similarity: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    per_support_distances: list[torch.Tensor] = []
    per_support_similarities: list[torch.Tensor] = []
    for support_index in range(support_patch_bank.shape[0]):
        support_rows = support_patch_bank[support_index].permute(1, 2, 0).reshape(-1, channels)
        support_rows = F.normalize(support_rows, dim=1, eps=1e-6)
        similarities = query_rows @ support_rows.t()
        k = max(1, min(int(knn_topk), similarities.shape[1]))
        if k == 1:
            similarity = similarities.max(dim=1).values
        else:
            similarity = similarities.topk(k=k, dim=1).values.mean(dim=1)
        per_support_similarities.append(similarity)
        per_support_distances.append(1.0 - similarity)
    stacked = torch.stack(per_support_distances, dim=1)
    consensus_k = max(1, min(int(support_consensus_topk), stacked.shape[1]))
    consensus_distance = stacked.topk(k=consensus_k, dim=1, largest=False).values.mean(dim=1)
    consensus_map = consensus_distance.view(batch_size, 1, height, width)
    if not return_best_normal_similarity:
        return consensus_map
    best_normal_similarity = torch.stack(per_support_similarities, dim=1).max(dim=1).values
    return consensus_map, best_normal_similarity.view(batch_size, 1, height, width)


def support_distance_stack(
    query_patch_map: torch.Tensor,
    support_patch_bank: torch.Tensor,
    knn_topk: int,
) -> tuple[torch.Tensor, int, int, int]:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    per_support_distances: list[torch.Tensor] = []
    for support_index in range(support_patch_bank.shape[0]):
        support_rows = support_patch_bank[support_index].permute(1, 2, 0).reshape(-1, channels)
        support_rows = F.normalize(support_rows, dim=1, eps=1e-6)
        similarities = query_rows @ support_rows.t()
        k = max(1, min(int(knn_topk), similarities.shape[1]))
        if k == 1:
            similarity = similarities.max(dim=1).values
        else:
            similarity = similarities.topk(k=k, dim=1).values.mean(dim=1)
        per_support_distances.append(1.0 - similarity)
    return torch.stack(per_support_distances, dim=1), batch_size, height, width


def consensus_distance_and_weight(
    distance_stack: torch.Tensor,
    support_consensus_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    consensus_k = max(1, min(int(support_consensus_topk), distance_stack.shape[1]))
    nearest_distances = distance_stack.topk(k=consensus_k, dim=1, largest=False).values
    consensus_distance = nearest_distances.mean(dim=1)
    dispersion = nearest_distances.std(dim=1, unbiased=False)
    support_weight = consensus_distance / (consensus_distance + dispersion + 1e-6)
    return consensus_distance, support_weight.clamp_(0.0, 1.0)


def patchcore_support_consensus_defect_boost_score_map(
    query_patch_map: torch.Tensor,
    support_patch_bank: torch.Tensor,
    defect_reference_bank: torch.Tensor,
    knn_topk: int,
    support_consensus_topk: int,
    defect_boost_weight: float,
) -> torch.Tensor:
    consensus_map, best_normal_similarity_map = patchcore_support_consensus_score_map(
        query_patch_map=query_patch_map,
        support_patch_bank=support_patch_bank,
        knn_topk=knn_topk,
        support_consensus_topk=support_consensus_topk,
        return_best_normal_similarity=True,
    )
    batch_size, _, height, width = consensus_map.shape
    channels = query_patch_map.shape[1]
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    defect_reference_bank = F.normalize(defect_reference_bank, dim=1, eps=1e-6)
    defect_similarities = query_rows @ defect_reference_bank.t()
    k_defect = max(1, min(int(knn_topk), defect_similarities.shape[1]))
    if k_defect == 1:
        defect_similarity = defect_similarities.max(dim=1).values
    else:
        defect_similarity = defect_similarities.topk(k=k_defect, dim=1).values.mean(dim=1)
    defect_similarity_map = defect_similarity.view(batch_size, 1, height, width)
    defect_margin = torch.relu(defect_similarity_map - best_normal_similarity_map)
    return consensus_map + float(defect_boost_weight) * defect_margin


def patchcore_match_reweight_score_map(
    query_patch_map: torch.Tensor,
    support_patch_bank: torch.Tensor,
    knn_topk: int,
    support_consensus_topk: int,
    match_reweight_alpha: float,
) -> torch.Tensor:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    per_support_distances: list[torch.Tensor] = []
    per_support_similarities: list[torch.Tensor] = []
    for support_index in range(support_patch_bank.shape[0]):
        support_rows = support_patch_bank[support_index].permute(1, 2, 0).reshape(-1, channels)
        support_rows = F.normalize(support_rows, dim=1, eps=1e-6)
        similarities = query_rows @ support_rows.t()
        k = max(1, min(int(knn_topk), similarities.shape[1]))
        if k == 1:
            similarity = similarities.max(dim=1).values
        else:
            similarity = similarities.topk(k=k, dim=1).values.mean(dim=1)
        per_support_similarities.append(similarity)
        per_support_distances.append(1.0 - similarity)
    distance_stack = torch.stack(per_support_distances, dim=1)
    similarity_stack = torch.stack(per_support_similarities, dim=1)
    consensus_k = max(1, min(int(support_consensus_topk), distance_stack.shape[1]))
    top_values, top_indices = similarity_stack.topk(k=consensus_k, dim=1)
    gathered_distances = distance_stack.gather(dim=1, index=top_indices)
    weights = torch.softmax(float(match_reweight_alpha) * top_values, dim=1)
    weighted_distance = (weights * gathered_distances).sum(dim=1)
    return weighted_distance.view(batch_size, 1, height, width)


def patchcore_supportaware_preproc_consensus_score_map(
    query_patch_map: torch.Tensor,
    object_normalized_query_patch_map: torch.Tensor,
    support_patch_bank: torch.Tensor,
    object_normalized_support_patch_bank: torch.Tensor,
    knn_topk: int,
    support_consensus_topk: int,
    preproc_agreement_beta: float,
) -> torch.Tensor:
    base_distance_stack, batch_size, height, width = support_distance_stack(
        query_patch_map=query_patch_map,
        support_patch_bank=support_patch_bank,
        knn_topk=knn_topk,
    )
    normalized_distance_stack, _, _, _ = support_distance_stack(
        query_patch_map=object_normalized_query_patch_map,
        support_patch_bank=object_normalized_support_patch_bank,
        knn_topk=knn_topk,
    )
    base_distance, base_support_weight = consensus_distance_and_weight(
        distance_stack=base_distance_stack,
        support_consensus_topk=support_consensus_topk,
    )
    normalized_distance, normalized_support_weight = consensus_distance_and_weight(
        distance_stack=normalized_distance_stack,
        support_consensus_topk=support_consensus_topk,
    )
    view_agreement = torch.exp(-float(preproc_agreement_beta) * torch.abs(base_distance - normalized_distance))
    support_agreement = torch.sqrt((base_support_weight * normalized_support_weight).clamp_min(0.0))
    anomaly = torch.minimum(base_distance, normalized_distance) * support_agreement * view_agreement
    return anomaly.view(batch_size, 1, height, width)


def fit_global_subspace(reference_bank: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if reference_bank.ndim != 2:
        raise ValueError(f"Expected a 2D reference bank, got {tuple(reference_bank.shape)}")
    centered = reference_bank - reference_bank.mean(dim=0, keepdim=True)
    rank_limit = min(centered.shape[0] - 1, centered.shape[1], int(max_dim))
    if rank_limit <= 0:
        basis = reference_bank.new_zeros((reference_bank.shape[1], 0))
        return reference_bank.mean(dim=0), basis
    work = centered.float()
    _, _, basis_t = torch.pca_lowrank(work, q=rank_limit, center=False)
    return reference_bank.mean(dim=0), basis_t[:, :rank_limit].contiguous().to(dtype=reference_bank.dtype, device=reference_bank.device)


def global_subspace_score_map(query_patch_map: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = query_patch_map.shape
    query_rows = F.normalize(query_patch_map, dim=1, eps=1e-6).permute(0, 2, 3, 1).reshape(-1, channels)
    centered = query_rows - mean.unsqueeze(0)
    if basis.numel() == 0:
        residual = centered
    else:
        projection = (centered @ basis) @ basis.transpose(0, 1)
        residual = centered - projection
    scores = residual.norm(dim=1)
    return scores.view(batch_size, 1, height, width)


def score_patch_batch(
    patch_batch: torch.Tensor,
    config: dict[str, object],
    reference_bank: torch.Tensor | None = None,
    defect_bank: torch.Tensor | None = None,
    reference_local_radii: torch.Tensor | None = None,
    support_patch_bank: torch.Tensor | None = None,
    object_normalized_patch_batch: torch.Tensor | None = None,
    object_normalized_support_patch_bank: torch.Tensor | None = None,
    subspace_mean: torch.Tensor | None = None,
    subspace_basis: torch.Tensor | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    if config["method"] == METHOD_PATCHCORE_KNN:
        anomaly_map = patchcore_knn_score_map(
            query_patch_map=patch_batch,
            reference_bank=reference_bank,
            knn_topk=int(config["knn_topk"]),
        )
    elif config["method"] == METHOD_PATCHCORE_DEFECT_BOOST:
        anomaly_map = patchcore_defect_boost_score_map(
            query_patch_map=patch_batch,
            normal_reference_bank=reference_bank,
            defect_reference_bank=defect_bank,
            knn_topk=int(config["knn_topk"]),
            defect_boost_weight=float(config["defect_boost_weight"]),
        )
    elif config["method"] == METHOD_PATCHCORE_LOCAL_SCALE:
        anomaly_map = patchcore_local_scale_score_map(
            query_patch_map=patch_batch,
            reference_bank=reference_bank,
            reference_local_radii=reference_local_radii,
            knn_topk=int(config["knn_topk"]),
        )
    elif config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS:
        anomaly_map = patchcore_support_consensus_score_map(
            query_patch_map=patch_batch,
            support_patch_bank=support_patch_bank,
            knn_topk=int(config["knn_topk"]),
            support_consensus_topk=int(config["support_consensus_topk"]),
        )
    elif config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST:
        anomaly_map = patchcore_support_consensus_defect_boost_score_map(
            query_patch_map=patch_batch,
            support_patch_bank=support_patch_bank,
            defect_reference_bank=defect_bank,
            knn_topk=int(config["knn_topk"]),
            support_consensus_topk=int(config["support_consensus_topk"]),
            defect_boost_weight=float(config["defect_boost_weight"]),
        )
    elif config["method"] == METHOD_PATCHCORE_MATCH_REWEIGHT:
        anomaly_map = patchcore_match_reweight_score_map(
            query_patch_map=patch_batch,
            support_patch_bank=support_patch_bank,
            knn_topk=int(config["knn_topk"]),
            support_consensus_topk=int(config["support_consensus_topk"]),
            match_reweight_alpha=float(config["match_reweight_alpha"]),
        )
    elif config["method"] == METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS:
        anomaly_map = patchcore_supportaware_preproc_consensus_score_map(
            query_patch_map=patch_batch,
            object_normalized_query_patch_map=object_normalized_patch_batch,
            support_patch_bank=support_patch_bank,
            object_normalized_support_patch_bank=object_normalized_support_patch_bank,
            knn_topk=int(config["knn_topk"]),
            support_consensus_topk=int(config["support_consensus_topk"]),
            preproc_agreement_beta=float(config["preproc_agreement_beta"]),
        )
    elif config["method"] == METHOD_GLOBAL_SUBSPACE:
        anomaly_map = global_subspace_score_map(
            query_patch_map=patch_batch,
            mean=subspace_mean,
            basis=subspace_basis,
        )
    else:
        raise ValueError(f"Unsupported method: {config['method']}")
    output_size = patch_batch_output_size(
        patch_batch=patch_batch,
        patch_multiple=int(config.get("resize_patch_multiple", 14)),
    )
    scored = score_map_outputs(
        score_map=anomaly_map,
        image_size=output_size,
        topk_ratio=float(config["topk_ratio"]),
        aggregation_mode=AGGREGATION_MODE_TOPK_MEAN,
        aggregation_stage="patch",
    )
    image_scores = scored["image_scores"].detach().cpu().tolist()
    patch_scores = [anomaly_map[index, 0].detach().cpu().numpy().astype(np.float32) for index in range(anomaly_map.shape[0])]
    return image_scores, patch_scores


def prepare_reference_variants(
    support_patch_batch: torch.Tensor,
    coreset_ratios: list[float],
    device: torch.device,
    seed: int,
) -> dict[float, torch.Tensor]:
    support_rows = flatten_patch_map(support_patch_batch.to(device))
    variants: dict[float, torch.Tensor] = {}
    for ratio in sorted({float(value) for value in coreset_ratios}, reverse=True):
        if ratio >= 0.999999:
            variants[ratio] = support_rows
            continue
        variants[ratio] = greedy_farthest_point_coreset(
            features=support_rows,
            keep_ratio=float(ratio),
            seed=int(seed),
        )
    return variants


def evaluate_split_scores(
    patch_batch_cpu: torch.Tensor,
    config: dict[str, object],
    output_size: int | tuple[int, int],
    device: torch.device,
    score_batch_size: int,
    reference_bank: torch.Tensor | None = None,
    defect_bank: torch.Tensor | None = None,
    reference_local_radii: torch.Tensor | None = None,
    support_patch_bank: torch.Tensor | None = None,
    object_normalized_patch_batch_cpu: torch.Tensor | None = None,
    object_normalized_support_patch_bank: torch.Tensor | None = None,
    patch_mask_cpu: torch.Tensor | None = None,
    subspace_mean: torch.Tensor | None = None,
    subspace_basis: torch.Tensor | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    image_scores: list[float] = []
    upsampled_maps: list[np.ndarray] = []
    for start in range(0, patch_batch_cpu.shape[0], int(score_batch_size)):
        end = min(patch_batch_cpu.shape[0], start + int(score_batch_size))
        patch_batch = patch_batch_cpu[start:end].to(device)
        object_normalized_patch_batch = None
        if object_normalized_patch_batch_cpu is not None:
            object_normalized_patch_batch = object_normalized_patch_batch_cpu[start:end].to(device)
        patch_mask_batch = None
        if patch_mask_cpu is not None:
            patch_mask_batch = patch_mask_cpu[start:end].to(device)
        if config["method"] == METHOD_PATCHCORE_KNN:
            anomaly_map = patchcore_knn_score_map(
                query_patch_map=patch_batch,
                reference_bank=reference_bank,
                knn_topk=int(config["knn_topk"]),
            )
        elif config["method"] == METHOD_PATCHCORE_DEFECT_BOOST:
            anomaly_map = patchcore_defect_boost_score_map(
                query_patch_map=patch_batch,
                normal_reference_bank=reference_bank,
                defect_reference_bank=defect_bank,
                knn_topk=int(config["knn_topk"]),
                defect_boost_weight=float(config["defect_boost_weight"]),
            )
        elif config["method"] == METHOD_PATCHCORE_LOCAL_SCALE:
            anomaly_map = patchcore_local_scale_score_map(
                query_patch_map=patch_batch,
                reference_bank=reference_bank,
                reference_local_radii=reference_local_radii,
                knn_topk=int(config["knn_topk"]),
            )
        elif config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS:
            anomaly_map = patchcore_support_consensus_score_map(
                query_patch_map=patch_batch,
                support_patch_bank=support_patch_bank,
                knn_topk=int(config["knn_topk"]),
                support_consensus_topk=int(config["support_consensus_topk"]),
            )
        elif config["method"] == METHOD_PATCHCORE_SUPPORT_CONSENSUS_DEFECT_BOOST:
            anomaly_map = patchcore_support_consensus_defect_boost_score_map(
                query_patch_map=patch_batch,
                support_patch_bank=support_patch_bank,
                defect_reference_bank=defect_bank,
                knn_topk=int(config["knn_topk"]),
                support_consensus_topk=int(config["support_consensus_topk"]),
                defect_boost_weight=float(config["defect_boost_weight"]),
            )
        elif config["method"] == METHOD_PATCHCORE_MATCH_REWEIGHT:
            anomaly_map = patchcore_match_reweight_score_map(
                query_patch_map=patch_batch,
                support_patch_bank=support_patch_bank,
                knn_topk=int(config["knn_topk"]),
                support_consensus_topk=int(config["support_consensus_topk"]),
                match_reweight_alpha=float(config["match_reweight_alpha"]),
            )
        elif config["method"] == METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS:
            anomaly_map = patchcore_supportaware_preproc_consensus_score_map(
                query_patch_map=patch_batch,
                object_normalized_query_patch_map=object_normalized_patch_batch,
                support_patch_bank=support_patch_bank,
                object_normalized_support_patch_bank=object_normalized_support_patch_bank,
                knn_topk=int(config["knn_topk"]),
                support_consensus_topk=int(config["support_consensus_topk"]),
                preproc_agreement_beta=float(config["preproc_agreement_beta"]),
            )
        else:
            anomaly_map = global_subspace_score_map(
                query_patch_map=patch_batch,
                mean=subspace_mean,
                basis=subspace_basis,
            )
        if patch_mask_batch is not None:
            anomaly_map = anomaly_map * patch_mask_batch.to(dtype=anomaly_map.dtype)
        scored = score_map_outputs(
            score_map=anomaly_map,
            image_size=output_size,
            topk_ratio=float(config["topk_ratio"]),
            aggregation_mode=AGGREGATION_MODE_TOPK_MEAN,
            aggregation_stage="patch",
        )
        image_scores.extend(scored["image_scores"].detach().cpu().tolist())
        for index in range(scored["upsampled_map"].shape[0]):
            upsampled_maps.append(scored["upsampled_map"][index, 0].detach().cpu().numpy().astype(np.float32))
    return image_scores, upsampled_maps


def evaluate_records(
    records: list[dict[str, object]],
    scores: list[float],
    threshold_map: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "scores": [], "thresholds": []})
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        grouped[category]["labels"].append(int(record["label"]))
        grouped[category]["scores"].append(float(score))
        grouped[category]["thresholds"].append(float(threshold_map[category]["threshold"]))

    per_category_rows: list[dict[str, object]] = []
    all_labels: list[int] = []
    all_scores: list[float] = []
    total_pred_positive = 0
    total_pred_negative = 0
    for category in sorted(grouped):
        labels = grouped[category]["labels"]
        category_scores = grouped[category]["scores"]
        threshold = float(threshold_map[category]["threshold"])
        cls_metrics = classification_metrics(labels=labels, scores=category_scores, threshold=threshold)
        row = {
            "category": category,
            "num_query_total": len(labels),
            "num_query_normal": int(sum(label == 0 for label in labels)),
            "num_query_defect": int(sum(label == 1 for label in labels)),
            "image_auroc": binary_auroc(labels, category_scores),
            "image_ap": average_precision_score(labels, category_scores),
            "accuracy": cls_metrics["accuracy"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "specificity": cls_metrics["specificity"],
            "balanced_accuracy": cls_metrics["balanced_accuracy"],
            "f1": cls_metrics["f1"],
            "num_pred_positive": cls_metrics["num_pred_positive"],
            "num_pred_negative": cls_metrics["num_pred_negative"],
            "tp": cls_metrics["tp"],
            "tn": cls_metrics["tn"],
            "fp": cls_metrics["fp"],
            "fn": cls_metrics["fn"],
            "threshold": threshold,
        }
        per_category_rows.append(row)
        all_labels.extend(labels)
        all_scores.extend(category_scores)
        total_pred_positive += int(cls_metrics["num_pred_positive"])
        total_pred_negative += int(cls_metrics["num_pred_negative"])

    prediction_rows: list[int] = []
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        prediction_rows.append(int(float(score) >= float(threshold_map[category]["threshold"])))
    labels_array = np.asarray(all_labels, dtype=np.int64)
    preds_array = np.asarray(prediction_rows, dtype=np.int64)
    tp = int(((preds_array == 1) & (labels_array == 1)).sum())
    tn = int(((preds_array == 0) & (labels_array == 0)).sum())
    fp = int(((preds_array == 1) & (labels_array == 0)).sum())
    fn = int(((preds_array == 0) & (labels_array == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    aggregate = {
        "num_test_images": len(records),
        "num_query_normal": int(sum(record["label"] == 0 for record in records)),
        "num_query_defect": int(sum(record["label"] == 1 for record in records)),
        "image_auroc_mean": float(np.mean([row["image_auroc"] for row in per_category_rows])),
        "image_ap_mean": float(np.mean([row["image_ap"] for row in per_category_rows])),
        "accuracy": float((tp + tn) / max(1, len(records))),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "num_pred_positive": int(preds_array.sum()),
        "num_pred_negative": int((1 - preds_array).sum()),
    }
    return per_category_rows, aggregate


def save_predictions(
    path: Path,
    records: list[dict[str, object]],
    scores: list[float],
    threshold_map: dict[str, dict[str, float]],
) -> None:
    rows: list[dict[str, object]] = []
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        threshold = float(threshold_map[category]["threshold"])
        rows.append(
            {
                "path": str(record["path"]),
                "category": category,
                "label": int(record["label"]),
                "defect_type": str(record["defect_type"]),
                "image_score": float(score),
                "pred_label": int(float(score) >= threshold),
                "threshold": threshold,
                "mask_path": str(record["mask_path"]),
                "role": str(record["role"]),
            }
        )
    write_csv(path, rows)


def detect_alerts(experiment_row: dict[str, object], per_category_rows: list[dict[str, object]]) -> list[str]:
    alerts: list[str] = []
    if not np.isfinite(float(experiment_row["image_auroc_mean"])):
        alerts.append("non_finite_image_auroc_mean")
    if not np.isfinite(float(experiment_row["image_ap_mean"])):
        alerts.append("non_finite_image_ap_mean")
    if int(experiment_row["num_pred_positive"]) == 0:
        alerts.append("degenerate_all_negative_predictions")
    if int(experiment_row["num_pred_negative"]) == 0:
        alerts.append("degenerate_all_positive_predictions")
    if float(experiment_row["balanced_accuracy"]) <= 0.500001 and (
        float(experiment_row["recall"]) <= 0.05 or float(experiment_row["specificity"]) <= 0.05
    ):
        alerts.append("threshold_metrics_collapsed")
    bottle_rows = [row for row in per_category_rows if row["category"] == CONTROL_CATEGORY]
    if bottle_rows and float(bottle_rows[0]["image_auroc"]) < 0.60:
        alerts.append("bottle_regression_severe")
    return alerts


def evaluate_category_candidate(
    category: str,
    manifest,
    support_train,
    support_holdout,
    output_root: Path,
    encoder: DinoV2PatchEncoder,
    config: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    feature_source = str(config["feature_source"])
    resize_mode = str(config.get("resize_mode", args.resize_mode))
    resize_patch_multiple = int(config.get("resize_patch_multiple", args.resize_patch_multiple))
    cache_variant = (
        f"multiview_{FEATURE_CACHE_VERSION}"
        if config["method"] == METHOD_PATCHCORE_SUPPORTAWARE_PREPROC_CONSENSUS
        else f"base_{FEATURE_CACHE_VERSION}"
    )
    size_root = Path(args.feature_cache_root) / slug(args.backbone)
    size_tag = f"img{int(args.image_size):03d}"
    if resize_mode == RESIZE_MODE_SQUARE and resize_patch_multiple == 14:
        category_cache_dir = size_root / size_tag / feature_source / category / cache_variant
    else:
        category_cache_dir = (
            size_root
            / f"{size_tag}_{resize_mode_tag(resize_mode=resize_mode, patch_multiple=resize_patch_multiple)}"
            / feature_source
            / category
            / cache_variant
        )
    all_paths = [str(sample.path) for sample in [*support_train, *support_holdout, *manifest.support_defect, *manifest.query_eval]]
    cache_start = time.time()
    written = populate_dinov2_feature_cache(
        encoder=encoder,
        image_paths=all_paths,
        cache_dir=category_cache_dir,
        image_size=args.image_size,
        resize_mode=resize_mode,
        patch_multiple=resize_patch_multiple,
        feature_source=feature_source,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    cache_seconds = time.time() - cache_start

    support_train_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in support_train],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_BASE,
    )
    holdout_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in support_holdout],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_BASE,
    )
    defect_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in manifest.support_defect],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_BASE,
    )
    query_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in manifest.query_eval],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_BASE,
    )
    support_train_object_normalized_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in support_train],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_OBJECT_NORMALIZED,
    )
    holdout_object_normalized_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in support_holdout],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_OBJECT_NORMALIZED,
    )
    defect_object_normalized_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in manifest.support_defect],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_OBJECT_NORMALIZED,
    )
    query_object_normalized_patch, _ = load_dinov2_feature_cache_batch(
        [sample.path for sample in manifest.query_eval],
        cache_dir=category_cache_dir,
        device=None,
        view=FEATURE_VIEW_OBJECT_NORMALIZED,
    )

    support_train_mask = foreground_mask_from_patch_batch(support_train_object_normalized_patch)
    holdout_mask = foreground_mask_from_patch_batch(holdout_object_normalized_patch)
    defect_mask = foreground_mask_from_patch_batch(defect_object_normalized_patch)
    query_mask = foreground_mask_from_patch_batch(query_object_normalized_patch)

    support_train_patch_aug, support_train_mask_aug = augment_patch_batch_with_rotations(
        support_train_patch,
        support_train_mask,
    )
    support_train_object_normalized_patch_aug = support_train_patch_aug * support_train_mask_aug.to(dtype=support_train_patch_aug.dtype)
    holdout_output_size = patch_batch_output_size(holdout_patch, resize_patch_multiple)
    defect_output_size = patch_batch_output_size(defect_patch, resize_patch_multiple)
    query_output_size = patch_batch_output_size(query_patch, resize_patch_multiple)

    reference_variants = prepare_reference_variants(
        support_patch_batch=support_train_object_normalized_patch_aug,
        coreset_ratios=args.coreset_ratios,
        device=device,
        seed=seed,
    )
    reference_bank = reference_variants[float(config["coreset_ratio"])]
    defect_bank = flatten_patch_map(defect_object_normalized_patch.to(device))
    reference_local_radii = None
    support_patch_bank = support_train_patch_aug.to(device)
    object_normalized_support_patch_bank = support_train_object_normalized_patch_aug.to(device)
    subspace_mean = None
    subspace_basis = None
    if config["method"] == METHOD_GLOBAL_SUBSPACE:
        mean, basis = fit_global_subspace(reference_bank, max_dim=int(config["subspace_dim"]))
        subspace_mean = mean
        subspace_basis = basis[:, : int(config["subspace_dim"])]
    elif config["method"] == METHOD_PATCHCORE_LOCAL_SCALE:
        reference_local_radii = compute_reference_local_radii(
            reference_bank=reference_bank,
            neighbor_topk=int(config["local_scale_neighbor_topk"]),
        )

    holdout_records = build_records(support_holdout, role="support_holdout") + build_records(manifest.support_defect, role="support_defect")
    holdout_scores_normal, _ = evaluate_split_scores(
        patch_batch_cpu=holdout_patch,
        config=config,
        output_size=holdout_output_size,
        device=device,
        score_batch_size=args.score_batch_size,
        reference_bank=reference_bank,
        defect_bank=defect_bank,
        reference_local_radii=reference_local_radii,
        support_patch_bank=support_patch_bank,
        object_normalized_patch_batch_cpu=holdout_object_normalized_patch,
        object_normalized_support_patch_bank=object_normalized_support_patch_bank,
        patch_mask_cpu=holdout_mask,
        subspace_mean=subspace_mean,
        subspace_basis=subspace_basis,
    )
    holdout_scores_defect, _ = evaluate_split_scores(
        patch_batch_cpu=defect_patch,
        config=config,
        output_size=defect_output_size,
        device=device,
        score_batch_size=args.score_batch_size,
        reference_bank=reference_bank,
        defect_bank=defect_bank,
        reference_local_radii=reference_local_radii,
        support_patch_bank=support_patch_bank,
        object_normalized_patch_batch_cpu=defect_object_normalized_patch,
        object_normalized_support_patch_bank=object_normalized_support_patch_bank,
        patch_mask_cpu=defect_mask,
        subspace_mean=subspace_mean,
        subspace_basis=subspace_basis,
    )
    holdout_scores = holdout_scores_normal + holdout_scores_defect
    threshold_map = choose_thresholds_by_category(records=holdout_records, scores=holdout_scores)

    query_records = build_records(manifest.query_eval, role="query_eval")
    query_scores, query_score_maps = evaluate_split_scores(
        patch_batch_cpu=query_patch,
        config=config,
        output_size=query_output_size,
        device=device,
        score_batch_size=args.score_batch_size,
        reference_bank=reference_bank,
        defect_bank=defect_bank,
        reference_local_radii=reference_local_radii,
        support_patch_bank=support_patch_bank,
        object_normalized_patch_batch_cpu=query_object_normalized_patch,
        object_normalized_support_patch_bank=object_normalized_support_patch_bank,
        patch_mask_cpu=query_mask,
        subspace_mean=subspace_mean,
        subspace_basis=subspace_basis,
    )
    per_category_rows, aggregate = evaluate_records(
        records=query_records,
        scores=query_scores,
        threshold_map=threshold_map,
    )
    masks = [load_mask_array(sample.mask_path, image_size=query_output_size) for sample in manifest.query_eval]
    pixel_auroc_value = pixel_auroc(masks, query_score_maps)
    pro_value = pro_score(masks, query_score_maps)
    category_row = dict(per_category_rows[0])
    category_row.update(
        {
            "method": str(config["method"]),
            "feature_source": feature_source,
            "candidate_name": candidate_name(args.backbone, config),
            "coreset_ratio": float(config["coreset_ratio"]),
            "knn_topk": int(config.get("knn_topk", 0)),
            "defect_boost_weight": float(config.get("defect_boost_weight", 0.0)),
            "local_scale_neighbor_topk": int(config.get("local_scale_neighbor_topk", 0)),
            "support_consensus_topk": int(config.get("support_consensus_topk", 0)),
            "match_reweight_alpha": float(config.get("match_reweight_alpha", 0.0)),
            "preproc_agreement_beta": float(config.get("preproc_agreement_beta", 0.0)),
            "subspace_dim": int(config.get("subspace_dim", 0)),
            "resize_mode": resize_mode,
            "resize_patch_multiple": resize_patch_multiple,
            "resolved_image_height": int(query_output_size[0]),
            "resolved_image_width": int(query_output_size[1]),
            "pixel_auroc": float(pixel_auroc_value),
            "pro": float(pro_value),
            "num_support_train": len(support_train),
            "num_support_holdout": len(support_holdout),
            "num_support_defect": len(manifest.support_defect),
            "feature_cache_dir": str(category_cache_dir),
            "feature_cache_written": len(written),
            "cache_seconds": float(cache_seconds),
            "support_threshold_source": "support_holdout_plus_support_defect_best_balanced_accuracy",
        }
    )

    category_dir = output_root / "candidates" / category_row["candidate_name"] / category
    category_dir.mkdir(parents=True, exist_ok=True)
    save_shared_split_manifest(manifest, category_dir / "split_manifest.json")
    write_json(category_dir / "category_threshold.json", threshold_map)
    write_json(category_dir / "metrics.json", category_row)
    write_json(
        category_dir / "config.json",
        {
            **config,
            "backbone": args.backbone,
            "image_size": args.image_size,
            "resize_mode": resize_mode,
            "resize_patch_multiple": resize_patch_multiple,
            "resolved_image_height": int(query_output_size[0]),
            "resolved_image_width": int(query_output_size[1]),
            "topk_ratio": args.topk_ratio,
            "holdout_fraction": args.holdout_fraction,
            "seed": seed,
        },
    )
    save_predictions(category_dir / "predictions.csv", query_records, query_scores, threshold_map)
    alerts = detect_alerts(
        experiment_row={
            "image_auroc_mean": category_row["image_auroc"],
            "image_ap_mean": category_row["image_ap"],
            "balanced_accuracy": category_row["balanced_accuracy"],
            "recall": category_row["recall"],
            "specificity": category_row["specificity"],
            "num_pred_positive": category_row["num_pred_positive"],
            "num_pred_negative": category_row["num_pred_negative"],
        },
        per_category_rows=[category_row],
    )
    write_json(category_dir / "alerts.json", {"alerts": alerts})
    return category_row, aggregate, alerts


def build_experiment_row(
    config: dict[str, object],
    candidate_label: str,
    subset: str,
    seed: int,
    per_category_rows: list[dict[str, object]],
    baseline_refs: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    all_scores_positive = 0
    all_scores_negative = 0
    for row in per_category_rows:
        all_scores_positive += int(row["num_pred_positive"])
        all_scores_negative += int(row["num_pred_negative"])
    total_images = sum(int(row["num_query_total"]) for row in per_category_rows)
    tp = sum(int(row["tp"]) for row in per_category_rows)
    tn = sum(int(row["tn"]) for row in per_category_rows)
    fp = sum(int(row["fp"]) for row in per_category_rows)
    fn = sum(int(row["fn"]) for row in per_category_rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    weak_rows = [row for row in per_category_rows if row["category"] in DEFAULT_WEAK5_CATEGORIES]
    bottle_row = next(row for row in per_category_rows if row["category"] == CONTROL_CATEGORY)
    experiment_row = {
        "experiment": candidate_label,
        "track": "pure-visual-dinov2",
        "scope": subset,
        "seed": int(seed),
        "num_categories": len(per_category_rows),
        "num_test_images": total_images,
        "num_query_normal": sum(int(row["num_query_normal"]) for row in per_category_rows),
        "num_query_defect": sum(int(row["num_query_defect"]) for row in per_category_rows),
        "image_auroc_mean": float(np.mean([float(row["image_auroc"]) for row in per_category_rows])),
        "image_ap_mean": float(np.mean([float(row["image_ap"]) for row in per_category_rows])),
        "pixel_auroc_mean": float(np.mean([float(row["pixel_auroc"]) for row in per_category_rows])),
        "pro_mean": float(np.mean([float(row["pro"]) for row in per_category_rows])),
        "accuracy": float((tp + tn) / max(1, total_images)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "num_pred_positive": int(all_scores_positive),
        "num_pred_negative": int(all_scores_negative),
        "weak5_image_auroc_mean": float(np.mean([float(row["image_auroc"]) for row in weak_rows])),
        "weak5_image_ap_mean": float(np.mean([float(row["image_ap"]) for row in weak_rows])),
        "bottle_image_auroc": float(bottle_row["image_auroc"]),
        "bottle_balanced_accuracy": float(bottle_row["balanced_accuracy"]),
        "method": str(config["method"]),
        "feature_source": str(config["feature_source"]),
        "coreset_ratio": float(config["coreset_ratio"]),
        "knn_topk": int(config.get("knn_topk", 0)),
        "defect_boost_weight": float(config.get("defect_boost_weight", 0.0)),
        "local_scale_neighbor_topk": int(config.get("local_scale_neighbor_topk", 0)),
        "support_consensus_topk": int(config.get("support_consensus_topk", 0)),
        "match_reweight_alpha": float(config.get("match_reweight_alpha", 0.0)),
        "preproc_agreement_beta": float(config.get("preproc_agreement_beta", 0.0)),
        "subspace_dim": int(config.get("subspace_dim", 0)),
        "resize_mode": str(config.get("resize_mode", RESIZE_MODE_SQUARE)),
        "resize_patch_multiple": int(config.get("resize_patch_multiple", 14)),
        "threshold_source": "per_category_support_holdout_plus_support_defect_best_balanced_accuracy",
    }
    e0 = baseline_refs["stage2_e0"]
    experiment_row["delta_vs_e0_auroc"] = float(experiment_row["image_auroc_mean"]) - float(e0["image_auroc_mean"])
    experiment_row["delta_vs_e0_ap"] = float(experiment_row["image_ap_mean"]) - float(e0["image_ap_mean"])
    experiment_row["delta_vs_e0_bottle_auroc"] = float(experiment_row["bottle_image_auroc"]) - float(
        baseline_refs["stage2_e0_per_category"][CONTROL_CATEGORY]["image_auroc"]
    )
    alerts = detect_alerts(experiment_row=experiment_row, per_category_rows=per_category_rows)
    experiment_row["alerts"] = ";".join(alerts)
    return experiment_row, alerts


def summary_lines(experiment_rows: list[dict[str, object]], baseline_refs: dict[str, object]) -> list[str]:
    lines = ["# Pure Visual DINOv2 Summary", ""]
    e0 = baseline_refs["stage2_e0"]
    incumbent = baseline_refs["stage4_incumbent"]
    lines.append(
        "E0: "
        f"auroc={float(e0['image_auroc_mean']):.6f} ap={float(e0['image_ap_mean']):.6f} "
        f"acc={float(e0['accuracy']):.6f} bal_acc={float(e0['balanced_accuracy']):.6f} "
        f"bottle={float(baseline_refs['stage2_e0_per_category'][CONTROL_CATEGORY]['image_auroc']):.6f}"
    )
    lines.append(
        "Stage4 incumbent: "
        f"auroc={float(incumbent['image_auroc_mean']):.6f} ap={float(incumbent['image_ap_mean']):.6f} "
        f"acc={float(incumbent['accuracy']):.6f} bal_acc={float(incumbent['balanced_accuracy']):.6f}"
    )
    lines.append("")
    for row in sorted(experiment_rows, key=lambda item: float(item["image_auroc_mean"]), reverse=True):
        lines.append(
            f"{row['experiment']}: auroc={float(row['image_auroc_mean']):.6f} ap={float(row['image_ap_mean']):.6f} "
            f"pix={float(row['pixel_auroc_mean']):.6f} pro={float(row['pro_mean']):.6f} "
            f"acc={float(row['accuracy']):.6f} bal_acc={float(row['balanced_accuracy']):.6f} "
            f"bottle={float(row['bottle_image_auroc']):.6f} alerts={row['alerts']}"
        )
    return lines


def main() -> None:
    args = parse_args()
    if args.categories is None:
        args.categories = list(default_categories(args.subset))
    if args.seeds is None:
        args.seeds = default_seeds(args.mode)
    if args.mode == "screening" and len(args.seeds) != 1:
        raise ValueError("screening mode expects exactly one seed.")

    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not args.skip_existing:
        raise RuntimeError(f"Official output directory is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_refs = load_baseline_refs(subset=args.subset)
    write_json(output_root / "baseline_refs.json", baseline_refs)
    write_json(
        output_root / "preflight_review.json",
        {
            "route": "pure-visual-dinov2",
            "mode": args.mode,
            "subset": args.subset,
            "seeds": args.seeds,
            "categories": args.categories,
            "backbone": args.backbone,
            "image_size": int(args.image_size),
            "resize_mode": str(args.resize_mode),
            "resize_patch_multiple": int(args.resize_patch_multiple),
            "risks": [
                "DINOv2 patch memory is substantially larger than the old CLIP cache.",
                "Retained baseline uses support-defect only for threshold calibration; support-aware variants add defect-bank retrieval boosts.",
                "Coreset ratios are intentionally small for runtime safety on dense DINO patches.",
            ],
            "go": True,
        },
    )

    device = resolve_device(args.device)
    encoder = DinoV2PatchEncoder(model_name=args.backbone, device=device)
    candidate_configs = build_candidate_configs(args)
    all_experiment_rows: list[dict[str, object]] = []
    all_per_category_rows: list[dict[str, object]] = []
    global_alerts: list[dict[str, object]] = []
    run_manifest: dict[str, object] = {
        "mode": args.mode,
        "subset": args.subset,
        "output_dir": str(output_root),
        "categories": args.categories,
        "seeds": args.seeds,
        "image_size": int(args.image_size),
        "resize_mode": str(args.resize_mode),
        "resize_patch_multiple": int(args.resize_patch_multiple),
        "num_candidates": len(candidate_configs),
        "candidates": [],
    }

    for seed in args.seeds:
        set_seed(int(seed))
        for config in candidate_configs:
            candidate_label = candidate_name(args.backbone, config)
            per_category_rows: list[dict[str, object]] = []
            candidate_alerts: list[str] = []
            for category in args.categories:
                manifest_path = ensure_manifest(
                    manifests_dir=Path(args.manifests_dir),
                    data_root=Path(args.data_root),
                    category=category,
                    support_normal_k=args.support_normal_k,
                    support_defect_k=args.support_defect_k,
                    seed=int(seed),
                )
                manifest = load_shared_split_manifest(manifest_path)
                support_train, support_holdout = split_support_holdout(
                    support_normal=manifest.support_normal,
                    holdout_fraction=args.holdout_fraction,
                    seed=int(seed),
                )
                category_row, _, alerts = evaluate_category_candidate(
                    category=category,
                    manifest=manifest,
                    support_train=support_train,
                    support_holdout=support_holdout,
                    output_root=output_root,
                    encoder=encoder,
                    config=config,
                    args=args,
                    device=device,
                    seed=int(seed),
                )
                category_row["experiment"] = candidate_label
                category_row["scope"] = args.subset
                category_row["seed"] = int(seed)
                per_category_rows.append(category_row)
                candidate_alerts.extend(alerts)
                torch.cuda.empty_cache()
            experiment_row, experiment_alerts = build_experiment_row(
                config=config,
                candidate_label=candidate_label,
                subset=args.subset,
                seed=int(seed),
                per_category_rows=per_category_rows,
                baseline_refs=baseline_refs,
            )
            all_experiment_rows.append(experiment_row)
            all_per_category_rows.extend(per_category_rows)
            global_alerts.append({"experiment": candidate_label, "seed": int(seed), "alerts": sorted(set(candidate_alerts + experiment_alerts))})
            run_manifest["candidates"].append(
                {
                    "seed": int(seed),
                    "experiment": candidate_label,
                    "config": config,
                }
            )
            write_json(output_root / "alerts.json", global_alerts)
            write_csv(output_root / "experiments.csv", all_experiment_rows)
            write_csv(output_root / "per_category.csv", all_per_category_rows)

    write_json(output_root / "run_manifest.json", run_manifest)
    (output_root / "summary.md").write_text("\n".join(summary_lines(all_experiment_rows, baseline_refs)) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_root), "num_experiments": len(all_experiment_rows)}, indent=2))


if __name__ == "__main__":
    main()
