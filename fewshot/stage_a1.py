from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .scoring import (
    AGGREGATION_MODE_TOPK_MEAN,
    AGGREGATION_STAGE_UPSAMPLED,
    FEATURE_LAYER_LOCAL,
    SCORE_MODE_ONE_MINUS_NORMAL,
    build_score_map,
    compute_similarity_maps,
    get_feature_map,
    score_map_outputs,
)


def binary_auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((1 - labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")

    order = np.argsort(-scores)
    labels = labels[order]
    tpr = np.cumsum(labels) / positive
    fpr = np.cumsum(1 - labels) / negative
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def pixel_auroc(mask_arrays: list[np.ndarray], score_maps: list[np.ndarray]) -> float:
    if not mask_arrays or not score_maps:
        return float("nan")
    flat_masks = np.concatenate([mask.reshape(-1) for mask in mask_arrays], axis=0)
    flat_scores = np.concatenate([score.reshape(-1) for score in score_maps], axis=0)
    return binary_auroc(flat_masks, flat_scores)


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    min_value = float(score_map.min())
    max_value = float(score_map.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - min_value) / (max_value - min_value)).astype(np.float32)


def heatmap_to_rgb(score_map: np.ndarray) -> np.ndarray:
    normalized = normalize_map(score_map)
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * normalized - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def make_overlay(image_rgb: np.ndarray, score_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = heatmap_to_rgb(score_map).astype(np.float32)
    base = image_rgb.astype(np.float32)
    overlay = (1.0 - alpha) * base + alpha * heatmap
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def save_prediction_artifacts(
    image_path: str | Path,
    image_rgb: np.ndarray,
    score_map: np.ndarray,
    output_dir: str | Path,
) -> dict[str, str]:
    image_path = Path(image_path)
    output_root = Path(output_dir)
    heatmap_dir = output_root / "heatmaps"
    overlay_dir = output_root / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = heatmap_dir / f"{image_path.stem}_heatmap.png"
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
    Image.fromarray(heatmap_to_rgb(score_map)).save(heatmap_path)
    Image.fromarray(make_overlay(image_rgb=image_rgb, score_map=score_map)).save(overlay_path)
    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }


@torch.no_grad()
def score_with_normal_prototype(
    encoder,
    images: torch.Tensor,
    normal_prototype: torch.Tensor,
    image_size: int,
    topk_ratio: float = 0.1,
    score_mode: str = SCORE_MODE_ONE_MINUS_NORMAL,
    aggregation_mode: str = AGGREGATION_MODE_TOPK_MEAN,
    aggregation_stage: str = AGGREGATION_STAGE_UPSAMPLED,
    feature_layer: str = FEATURE_LAYER_LOCAL,
    reference_topk: int = 1,
) -> dict[str, torch.Tensor]:
    encoded = encoder(images)
    feature_map = get_feature_map(encoded, feature_layer=feature_layer)
    similarity_maps = compute_similarity_maps(
        feature_map=feature_map,
        normal_prototype=normal_prototype,
        defect_prototype=None,
        reference_topk=reference_topk,
    )
    normal_map = similarity_maps["normal_map"]
    anomaly_map = build_score_map(similarity_maps, score_mode=score_mode)
    scored = score_map_outputs(
        anomaly_map,
        image_size=image_size,
        topk_ratio=topk_ratio,
        aggregation_mode=aggregation_mode,
        aggregation_stage=aggregation_stage,
    )
    return {
        "normal_map": normal_map,
        "anomaly_map": anomaly_map,
        "patch_map": scored["patch_map"],
        "upsampled_map": scored["upsampled_map"],
        "image_scores": scored["image_scores"],
    }
