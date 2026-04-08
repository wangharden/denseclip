import torch
import torch.nn.functional as F

from .coreset import greedy_farthest_point_coreset
from .feature_bank import flatten_feature_map
from .subspace import _fit_linear_subspace, _project_residual_norm


@torch.no_grad()
def coreset_subspace_score_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    subspace_dim: int,
    coreset_ratio: float = 0.25,
    seed: int = 42,
) -> torch.Tensor:
    if support_feature_batch.ndim != 4 or query_feature_batch.ndim != 4:
        raise ValueError(
            "coreset_subspace_score_map expects support/query in [N,C,H,W], "
            f"got {tuple(support_feature_batch.shape)} and {tuple(query_feature_batch.shape)}"
        )
    if support_feature_batch.shape[1:] != query_feature_batch.shape[1:]:
        raise ValueError(
            "Support/query feature batches must share channel and spatial shape, "
            f"but got {tuple(support_feature_batch.shape[1:])} and {tuple(query_feature_batch.shape[1:])}"
        )
    support = F.normalize(support_feature_batch, dim=1, eps=1e-6)
    query = F.normalize(query_feature_batch, dim=1, eps=1e-6)
    _, _, height, width = query.shape
    support_rows = flatten_feature_map(support)
    coreset_rows = greedy_farthest_point_coreset(
        features=support_rows,
        keep_ratio=float(coreset_ratio),
        seed=int(seed),
    )
    mean, basis = _fit_linear_subspace(coreset_rows, subspace_dim=int(subspace_dim))
    query_rows = query.permute(0, 2, 3, 1).reshape(-1, query.shape[1])
    scores = _project_residual_norm(query_rows, mean=mean, basis=basis)
    return scores.view(query.shape[0], 1, height, width)


__all__ = ["coreset_subspace_score_map"]
