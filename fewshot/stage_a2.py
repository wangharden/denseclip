import torch

from .scoring import (
    AGGREGATION_MODE_TOPK_MEAN,
    AGGREGATION_STAGE_UPSAMPLED,
    FEATURE_LAYER_LOCAL,
    build_score_map,
    compute_similarity_maps,
    get_feature_map,
    score_map_outputs,
)


@torch.no_grad()
def score_with_dual_prototype(
    encoder,
    images: torch.Tensor,
    normal_prototype: torch.Tensor,
    defect_prototype: torch.Tensor,
    image_size: int,
    topk_ratio: float = 0.1,
    score_mode: str = "defect-minus-normal",
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
        defect_prototype=defect_prototype,
        reference_topk=reference_topk,
    )
    anomaly_map = build_score_map(similarity_maps, score_mode=score_mode)
    scored = score_map_outputs(
        anomaly_map,
        image_size=image_size,
        topk_ratio=topk_ratio,
        aggregation_mode=aggregation_mode,
        aggregation_stage=aggregation_stage,
    )
    return {
        "normal_map": similarity_maps["normal_map"],
        "defect_map": similarity_maps["defect_map"],
        "contrast_map": similarity_maps["contrast_map"],
        "anomaly_map": anomaly_map,
        "patch_map": scored["patch_map"],
        "upsampled_map": scored["upsampled_map"],
        "image_scores": scored["image_scores"],
    }
