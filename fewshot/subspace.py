import torch
import torch.nn.functional as F


SUBSPACE_MODE_GLOBAL = "global"
SUBSPACE_MODE_LOCAL = "local"
SUBSPACE_MODES = (
    SUBSPACE_MODE_GLOBAL,
    SUBSPACE_MODE_LOCAL,
)


def _normalize_feature_batch(feature_batch: torch.Tensor) -> torch.Tensor:
    if feature_batch.ndim != 4:
        raise ValueError(f"Expected a 4D feature batch, got shape {tuple(feature_batch.shape)}")
    if feature_batch.shape[0] == 0:
        raise ValueError("Expected a non-empty feature batch.")
    return F.normalize(feature_batch, dim=1, eps=1e-6)


def _validate_feature_batches(support_feature_batch: torch.Tensor, query_feature_batch: torch.Tensor) -> None:
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


def _fit_linear_subspace(samples: torch.Tensor, subspace_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2D sample matrix, got shape {tuple(samples.shape)}")
    if samples.shape[0] == 0:
        raise ValueError("Cannot fit a subspace to an empty sample matrix.")

    mean = samples.mean(dim=0, keepdim=True)
    centered = samples - mean
    max_rank = min(centered.shape[0] - 1, centered.shape[1])
    if max_rank <= 0 or subspace_dim <= 0:
        basis = centered.new_zeros((centered.shape[1], 0))
        return mean.squeeze(0), basis

    rank = min(int(subspace_dim), max_rank)
    work_centered = centered
    if centered.dtype not in (torch.float32, torch.float64):
        work_centered = centered.float()
    _, _, vh = torch.linalg.svd(work_centered, full_matrices=False)
    basis = vh[:rank].transpose(0, 1).contiguous().to(dtype=centered.dtype, device=centered.device)
    return mean.squeeze(0), basis


def _project_residual_norm(samples: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = samples - mean.unsqueeze(0)
    if basis.numel() == 0:
        residual = centered
    else:
        projection = (centered @ basis) @ basis.transpose(0, 1)
        residual = centered - projection
    return residual.norm(dim=1)


def global_subspace_score_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    subspace_dim: int,
) -> torch.Tensor:
    _validate_feature_batches(support_feature_batch, query_feature_batch)
    support_feature_batch = _normalize_feature_batch(support_feature_batch)
    query_feature_batch = _normalize_feature_batch(query_feature_batch)
    _, channels, height, width = support_feature_batch.shape
    support_samples = support_feature_batch.permute(0, 2, 3, 1).reshape(-1, channels)
    query_samples = query_feature_batch.permute(0, 2, 3, 1).reshape(-1, channels)
    mean, basis = _fit_linear_subspace(support_samples, subspace_dim=subspace_dim)
    scores = _project_residual_norm(query_samples, mean=mean, basis=basis)
    return scores.view(query_feature_batch.shape[0], 1, height, width)


def local_subspace_score_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    subspace_dim: int,
) -> torch.Tensor:
    _validate_feature_batches(support_feature_batch, query_feature_batch)
    support_feature_batch = _normalize_feature_batch(support_feature_batch)
    query_feature_batch = _normalize_feature_batch(query_feature_batch)
    support_samples = support_feature_batch.permute(2, 3, 0, 1).contiguous()
    query_samples = query_feature_batch.permute(2, 3, 0, 1).contiguous()
    height, width, _, _ = support_samples.shape
    score_map = query_feature_batch.new_empty((query_feature_batch.shape[0], 1, height, width))

    for h in range(height):
        for w in range(width):
            mean, basis = _fit_linear_subspace(support_samples[h, w], subspace_dim=subspace_dim)
            patch_scores = _project_residual_norm(query_samples[h, w], mean=mean, basis=basis)
            score_map[:, 0, h, w] = patch_scores
    return score_map


def subspace_score_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    subspace_dim: int,
    mode: str = SUBSPACE_MODE_LOCAL,
) -> torch.Tensor:
    if mode == SUBSPACE_MODE_GLOBAL:
        return global_subspace_score_map(
            support_feature_batch=support_feature_batch,
            query_feature_batch=query_feature_batch,
            subspace_dim=subspace_dim,
        )
    if mode == SUBSPACE_MODE_LOCAL:
        return local_subspace_score_map(
            support_feature_batch=support_feature_batch,
            query_feature_batch=query_feature_batch,
            subspace_dim=subspace_dim,
        )
    raise ValueError(f"Unsupported subspace mode: {mode}")
