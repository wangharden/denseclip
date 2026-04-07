import torch
import torch.nn.functional as F


def _normalize_feature_batch(feature_batch: torch.Tensor) -> torch.Tensor:
    if feature_batch.ndim != 4:
        raise ValueError(f"Expected a 4D feature batch, got shape {tuple(feature_batch.shape)}")
    return F.normalize(feature_batch, dim=1)


def _validate_matching_inputs(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
) -> tuple[int, int, int]:
    if support_feature_batch.ndim != 4 or query_feature_batch.ndim != 4:
        raise ValueError(
            "coordinate_matching_similarity_map expects "
            f"support/query in [N,C,H,W], got {tuple(support_feature_batch.shape)} and {tuple(query_feature_batch.shape)}"
        )

    if support_feature_batch.shape[0] <= 0:
        raise ValueError("support_feature_batch must contain at least one support map")

    support_channels = support_feature_batch.shape[1]
    query_channels = query_feature_batch.shape[1]
    if support_channels != query_channels:
        raise ValueError(f"Channel mismatch: support has {support_channels}, query has {query_channels}")

    support_height, support_width = support_feature_batch.shape[2:]
    query_height, query_width = query_feature_batch.shape[2:]
    if (support_height, support_width) != (query_height, query_width):
        raise ValueError(
            "coordinate-aware matching expects aligned spatial sizes, got "
            f"support {(support_height, support_width)} and query {(query_height, query_width)}"
        )
    return support_channels, query_height, query_width


@torch.no_grad()
def coordinate_matching_similarity_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    match_k: int = 1,
    spatial_window: int = 0,
) -> torch.Tensor:
    channels, height, width = _validate_matching_inputs(support_feature_batch, query_feature_batch)
    support = _normalize_feature_batch(support_feature_batch)
    query = _normalize_feature_batch(query_feature_batch)
    batch = query.shape[0]
    similarity_map = torch.empty((batch, 1, height, width), device=query.device, dtype=query.dtype)
    topk = max(1, int(match_k))
    window = max(0, int(spatial_window))

    for h_index in range(height):
        h_start = max(0, h_index - window)
        h_end = min(height, h_index + window + 1)
        for w_index in range(width):
            w_start = max(0, w_index - window)
            w_end = min(width, w_index + window + 1)
            support_window = support[:, :, h_start:h_end, w_start:w_end]
            support_rows = support_window.permute(0, 2, 3, 1).reshape(-1, channels)
            query_rows = query[:, :, h_index, w_index]
            similarities = query_rows @ support_rows.t()
            k = min(topk, similarities.shape[1])
            if k == 1:
                reduced = similarities.max(dim=1).values
            else:
                reduced = similarities.topk(k=k, dim=1).values.mean(dim=1)
            similarity_map[:, 0, h_index, w_index] = reduced
    return similarity_map


@torch.no_grad()
def correspondence_similarity_map(
    support_feature_batch: torch.Tensor,
    query_feature_batch: torch.Tensor,
    match_k: int = 1,
    spatial_window: int = 0,
) -> torch.Tensor:
    return coordinate_matching_similarity_map(
        support_feature_batch=support_feature_batch,
        query_feature_batch=query_feature_batch,
        match_k=match_k,
        spatial_window=spatial_window,
    )


__all__ = [
    "coordinate_matching_similarity_map",
    "correspondence_similarity_map",
]
