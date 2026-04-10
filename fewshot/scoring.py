import torch
import torch.nn.functional as F

SCORE_MODE_ONE_MINUS_NORMAL = "one-minus-normal"
SCORE_MODE_NEG_NORMAL = "neg-normal"
SCORE_MODE_DEFECT_MINUS_NORMAL = "defect-minus-normal"
SCORE_MODE_NORMAL_MINUS_DEFECT = "normal-minus-defect"
SCORE_MODE_BLEND = "blend"
SCORE_MODES = (
    SCORE_MODE_ONE_MINUS_NORMAL,
    SCORE_MODE_NEG_NORMAL,
    SCORE_MODE_DEFECT_MINUS_NORMAL,
    SCORE_MODE_NORMAL_MINUS_DEFECT,
    SCORE_MODE_BLEND,
)
AGGREGATION_MODE_TOPK_MEAN = "topk_mean"
AGGREGATION_MODE_MAX = "max"
AGGREGATION_MODES = (
    AGGREGATION_MODE_TOPK_MEAN,
    AGGREGATION_MODE_MAX,
)
AGGREGATION_STAGE_PATCH = "patch"
AGGREGATION_STAGE_UPSAMPLED = "upsampled"
AGGREGATION_STAGES = (
    AGGREGATION_STAGE_PATCH,
    AGGREGATION_STAGE_UPSAMPLED,
)
FEATURE_LAYER_LOCAL = "local"
FEATURE_LAYER_LAYER4 = "layer4"
FEATURE_LAYERS = (
    FEATURE_LAYER_LOCAL,
    FEATURE_LAYER_LAYER4,
)


def topk_mean(score_map: torch.Tensor, topk_ratio: float = 0.1) -> torch.Tensor:
    batch_size = score_map.shape[0]
    flat = score_map.reshape(batch_size, -1)
    k = max(1, int(flat.shape[1] * topk_ratio))
    values, _ = torch.topk(flat, k=k, dim=1)
    return values.mean(dim=1)


def _normalize_spatial_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        height, width = size
        return int(height), int(width)
    scalar = int(size)
    return scalar, scalar


def upsample_score_map(score_map: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    return F.interpolate(score_map, size=_normalize_spatial_size(size), mode="bilinear", align_corners=False)


def aggregate_image_score(
    score_map: torch.Tensor,
    aggregation_mode: str = AGGREGATION_MODE_TOPK_MEAN,
    topk_ratio: float = 0.1,
) -> torch.Tensor:
    if aggregation_mode == AGGREGATION_MODE_TOPK_MEAN:
        return topk_mean(score_map, topk_ratio=topk_ratio)
    if aggregation_mode == AGGREGATION_MODE_MAX:
        batch_size = score_map.shape[0]
        return score_map.reshape(batch_size, -1).max(dim=1).values
    raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")


def get_feature_map(
    encoded_features: dict[str, torch.Tensor],
    feature_layer: str = FEATURE_LAYER_LOCAL,
) -> torch.Tensor:
    if feature_layer not in encoded_features:
        raise ValueError(f"Unsupported feature_layer: {feature_layer}")
    feature_map = encoded_features[feature_layer]
    if feature_map.ndim != 4:
        raise ValueError(f"Feature layer must be a 4D map, got {feature_layer} with shape {tuple(feature_map.shape)}")
    return feature_map


def _prepare_reference_bank(reference_bank: torch.Tensor) -> torch.Tensor:
    if reference_bank.ndim == 1:
        reference_bank = reference_bank.unsqueeze(0)
    if reference_bank.ndim != 2:
        raise ValueError(f"Reference bank must be a 1D or 2D tensor, got shape {tuple(reference_bank.shape)}")
    return F.normalize(reference_bank, dim=1).contiguous()


def reference_similarity_map(
    feature_map: torch.Tensor,
    reference_bank: torch.Tensor,
    reference_topk: int = 1,
) -> torch.Tensor:
    reference_bank = _prepare_reference_bank(reference_bank)
    feature_map = F.normalize(feature_map, dim=1)
    batch_size, channels, height, width = feature_map.shape
    flat_features = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)
    similarities = flat_features @ reference_bank.t()
    k = max(1, min(int(reference_topk), similarities.shape[1]))
    if k == 1:
        reduced_similarity = similarities.max(dim=1).values
    else:
        reduced_similarity = similarities.topk(k=k, dim=1).values.mean(dim=1)
    return reduced_similarity.view(batch_size, 1, height, width)


def compute_similarity_maps(
    feature_map: torch.Tensor | None = None,
    normal_prototype: torch.Tensor | None = None,
    defect_prototype: torch.Tensor | None = None,
    local_features: torch.Tensor | None = None,
    reference_topk: int = 1,
) -> dict[str, torch.Tensor]:
    if feature_map is None:
        feature_map = local_features
    if feature_map is None:
        raise ValueError("compute_similarity_maps requires feature_map or local_features.")
    if normal_prototype is None:
        raise ValueError("compute_similarity_maps requires normal_prototype.")
    normal_map = reference_similarity_map(feature_map, normal_prototype, reference_topk=reference_topk)

    if defect_prototype is not None and defect_prototype.numel() > 0:
        defect_map = reference_similarity_map(feature_map, defect_prototype, reference_topk=reference_topk)
    else:
        defect_map = -normal_map

    contrast_map = defect_map - normal_map
    return {
        "normal_map": normal_map,
        "defect_map": defect_map,
        "contrast_map": contrast_map,
    }


def build_score_map(
    similarity_maps: dict[str, torch.Tensor],
    score_mode: str = SCORE_MODE_ONE_MINUS_NORMAL,
) -> torch.Tensor:
    normal_map = similarity_maps["normal_map"]
    defect_map = similarity_maps["defect_map"]

    if score_mode == SCORE_MODE_ONE_MINUS_NORMAL:
        return 1.0 - normal_map
    if score_mode == SCORE_MODE_NEG_NORMAL:
        return -normal_map
    if score_mode == SCORE_MODE_DEFECT_MINUS_NORMAL:
        return defect_map - normal_map
    if score_mode == SCORE_MODE_NORMAL_MINUS_DEFECT:
        return normal_map - defect_map
    if score_mode == SCORE_MODE_BLEND:
        return 0.5 * (1.0 - normal_map) + 0.5 * defect_map
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def score_map_outputs(
    score_map: torch.Tensor,
    image_size: int | tuple[int, int],
    topk_ratio: float = 0.1,
    aggregation_mode: str = AGGREGATION_MODE_TOPK_MEAN,
    aggregation_stage: str = AGGREGATION_STAGE_UPSAMPLED,
) -> dict[str, torch.Tensor]:
    upsampled_map = upsample_score_map(score_map, size=image_size)
    if aggregation_stage == AGGREGATION_STAGE_PATCH:
        aggregation_map = score_map
    elif aggregation_stage == AGGREGATION_STAGE_UPSAMPLED:
        aggregation_map = upsampled_map
    else:
        raise ValueError(f"Unsupported aggregation_stage: {aggregation_stage}")
    image_scores = aggregate_image_score(
        aggregation_map,
        aggregation_mode=aggregation_mode,
        topk_ratio=topk_ratio,
    )
    return {
        "patch_map": score_map,
        "upsampled_map": upsampled_map,
        "image_scores": image_scores,
    }


def logits_to_score_outputs(
    score_logits: torch.Tensor,
    image_size: int | tuple[int, int],
    topk_ratio: float = 0.1,
) -> dict[str, torch.Tensor]:
    upsampled_logits = upsample_score_map(score_logits, size=image_size)
    image_logits = topk_mean(upsampled_logits, topk_ratio=topk_ratio)
    upsampled_scores = torch.sigmoid(upsampled_logits)
    image_scores = torch.sigmoid(image_logits)
    return {
        "upsampled_logits": upsampled_logits,
        "image_logits": image_logits,
        "upsampled_scores": upsampled_scores,
        "image_scores": image_scores,
    }
