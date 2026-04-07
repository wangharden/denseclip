import torch
import torch.nn.functional as F

from .feature_bank import flatten_feature_map
from .scoring import reference_similarity_map


def _clamp_ratio(refine_ratio: float) -> float:
    return min(max(float(refine_ratio), 0.0), 1.0)


def _clamp_alpha(blend_alpha: float) -> float:
    return min(max(float(blend_alpha), 0.0), 1.0)


def _support_anchor(support_reference: torch.Tensor) -> torch.Tensor:
    return F.normalize(support_reference.mean(dim=0, keepdim=True), dim=1)


def _select_pseudo_normal_patches(
    query_feature_map: torch.Tensor,
    normal_similarity_map: torch.Tensor,
    refine_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_features = flatten_feature_map(query_feature_map.unsqueeze(0))
    flat_scores = normal_similarity_map.reshape(-1)
    num_patches = flat_scores.shape[0]
    keep = max(1, int(round(num_patches * _clamp_ratio(refine_ratio))))
    keep = min(keep, num_patches)
    topk = flat_scores.topk(k=keep, dim=0)
    selected_features = flat_features[topk.indices]
    selected_scores = topk.values
    return selected_features, selected_scores


def _build_pseudo_normal_reference(
    query_feature_map: torch.Tensor,
    normal_similarity_map: torch.Tensor,
    refine_ratio: float,
) -> torch.Tensor:
    pseudo_features, pseudo_scores = _select_pseudo_normal_patches(
        query_feature_map=query_feature_map,
        normal_similarity_map=normal_similarity_map,
        refine_ratio=refine_ratio,
    )
    weights = torch.softmax(pseudo_scores, dim=0).unsqueeze(1)
    pseudo_reference = (pseudo_features * weights).sum(dim=0, keepdim=True)
    return F.normalize(pseudo_reference, dim=1)


def refine_normal_reference(
    support_reference: torch.Tensor,
    query_feature_map: torch.Tensor,
    reference_topk: int = 3,
    refine_ratio: float = 0.2,
    blend_alpha: float = 0.5,
    refine_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_reference = F.normalize(support_reference, dim=1)
    base_reference = support_reference
    support_mean = _support_anchor(support_reference)
    refined_reference = support_mean
    initial_normal_map = reference_similarity_map(
        feature_map=query_feature_map.unsqueeze(0),
        reference_bank=base_reference,
        reference_topk=reference_topk,
    )

    current_reference = base_reference
    current_normal_map = initial_normal_map
    alpha = _clamp_alpha(blend_alpha)

    for _ in range(max(1, int(refine_steps))):
        pseudo_reference = _build_pseudo_normal_reference(
            query_feature_map=query_feature_map,
            normal_similarity_map=current_normal_map,
            refine_ratio=refine_ratio,
        )
        refined_reference = F.normalize(
            (1.0 - alpha) * support_mean + alpha * pseudo_reference,
            dim=1,
        )
        current_reference = torch.cat([base_reference, refined_reference], dim=0)
        current_normal_map = reference_similarity_map(
            feature_map=query_feature_map.unsqueeze(0),
            reference_bank=current_reference,
            reference_topk=reference_topk,
        )
    return refined_reference, current_normal_map


def fastref_lite_normal_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    reference_topk: int = 3,
    refine_ratio: float = 0.2,
    blend_alpha: float = 0.5,
    refine_steps: int = 1,
) -> torch.Tensor:
    support_reference = flatten_feature_map(support_feature_batch)
    outputs: list[torch.Tensor] = []
    for index in range(query_feature_batch.shape[0]):
        query_feature_map = query_feature_batch[index]
        _, refined_normal_map = refine_normal_reference(
            support_reference=support_reference,
            query_feature_map=query_feature_map,
            reference_topk=reference_topk,
            refine_ratio=refine_ratio,
            blend_alpha=blend_alpha,
            refine_steps=refine_steps,
        )
        outputs.append(refined_normal_map)
    return torch.cat(outputs, dim=0)


__all__ = [
    "fastref_lite_normal_map",
    "refine_normal_reference",
]
