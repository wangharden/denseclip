import torch
import torch.nn.functional as F

from .subspace import _fit_linear_subspace, _project_residual_norm


def _normalize_feature_batch(feature_batch: torch.Tensor) -> torch.Tensor:
    if feature_batch.ndim != 4:
        raise ValueError(f"Expected a 4D feature batch, got shape {tuple(feature_batch.shape)}")
    if feature_batch.shape[0] <= 0:
        raise ValueError("Expected a non-empty feature batch.")
    return F.normalize(feature_batch, dim=1, eps=1e-6)


def _validate_feature_batches(support_feature_batch: torch.Tensor, query_feature_batch: torch.Tensor) -> tuple[int, int, int]:
    if support_feature_batch.ndim != 4 or query_feature_batch.ndim != 4:
        raise ValueError(
            "Support/query feature batches must both be 4D tensors "
            f"but got {tuple(support_feature_batch.shape)} and {tuple(query_feature_batch.shape)}"
        )
    if support_feature_batch.shape[0] == 0 or query_feature_batch.shape[0] == 0:
        raise ValueError("Support/query feature batches must both be non-empty.")
    if support_feature_batch.shape[1:] != query_feature_batch.shape[1:]:
        raise ValueError(
            "Support/query feature batches must share channel and spatial shape, "
            f"but got {tuple(support_feature_batch.shape[1:])} and {tuple(query_feature_batch.shape[1:])}"
        )
    _, channels, height, width = support_feature_batch.shape
    return channels, height, width


@torch.no_grad()
def retrieved_subspace_score_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    subspace_dim: int,
    retrieval_topk: int = 8,
    spatial_window: int = 0,
) -> torch.Tensor:
    _, height, width = _validate_feature_batches(support_feature_batch, query_feature_batch)
    support = _normalize_feature_batch(support_feature_batch)
    query = _normalize_feature_batch(query_feature_batch)
    batch = query.shape[0]
    topk = max(1, int(retrieval_topk))
    window = max(0, int(spatial_window))
    score_map = query.new_empty((batch, 1, height, width))

    for h_index in range(height):
        h_start = max(0, h_index - window)
        h_end = min(height, h_index + window + 1)
        for w_index in range(width):
            w_start = max(0, w_index - window)
            w_end = min(width, w_index + window + 1)
            support_rows = support[:, :, h_start:h_end, w_start:w_end].permute(0, 2, 3, 1).reshape(-1, support.shape[1])
            query_rows = query[:, :, h_index, w_index]
            similarities = query_rows @ support_rows.t()
            keep = min(topk, similarities.shape[1])
            selected_indices = similarities.topk(k=keep, dim=1).indices

            patch_scores: list[torch.Tensor] = []
            for query_index in range(batch):
                selected_rows = support_rows[selected_indices[query_index]]
                mean, basis = _fit_linear_subspace(selected_rows, subspace_dim=subspace_dim)
                residual = _project_residual_norm(query_rows[query_index : query_index + 1], mean=mean, basis=basis)
                patch_scores.append(residual)
            score_map[:, 0, h_index, w_index] = torch.cat(patch_scores, dim=0)
    return score_map


__all__ = ["retrieved_subspace_score_map"]
