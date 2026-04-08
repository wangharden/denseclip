import torch
import torch.nn.functional as F


def _normalize_rows(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(features.shape)}")
    if features.shape[0] <= 0:
        raise ValueError("Expected a non-empty feature tensor.")
    return F.normalize(features, dim=1, eps=1e-6)


@torch.no_grad()
def greedy_farthest_point_coreset(
    features: torch.Tensor,
    keep_ratio: float = 0.25,
    seed: int = 42,
) -> torch.Tensor:
    features = _normalize_rows(features)
    num_features = features.shape[0]
    keep_count = max(1, min(num_features, int(round(num_features * float(keep_ratio)))))
    if keep_count >= num_features:
        return features.contiguous()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    first_index = int(torch.randint(0, num_features, (1,), generator=generator).item())
    selected_indices = [first_index]
    min_distance = 1.0 - (features @ features[first_index : first_index + 1].t()).squeeze(1)
    min_distance[first_index] = -1.0

    while len(selected_indices) < keep_count:
        next_index = int(min_distance.argmax().item())
        selected_indices.append(next_index)
        next_distance = 1.0 - (features @ features[next_index : next_index + 1].t()).squeeze(1)
        min_distance = torch.minimum(min_distance, next_distance)
        min_distance[selected_indices] = -1.0
    index_tensor = torch.tensor(selected_indices, device=features.device, dtype=torch.long)
    return features.index_select(0, index_tensor).contiguous()


__all__ = ["greedy_farthest_point_coreset"]
