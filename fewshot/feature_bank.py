from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .data import ImageSample, make_support_batch


PROTOTYPE_FAMILY_MEAN = "mean"
PROTOTYPE_FAMILY_MEMORY_BANK = "memory_bank"
PROTOTYPE_FAMILY_MULTI_PROTOTYPE = "multi_prototype"
PROTOTYPE_FAMILIES = (
    PROTOTYPE_FAMILY_MEAN,
    PROTOTYPE_FAMILY_MEMORY_BANK,
    PROTOTYPE_FAMILY_MULTI_PROTOTYPE,
)


@dataclass
class PrototypeBank:
    normal_reference: torch.Tensor
    defect_reference: torch.Tensor | None
    normal_support_paths: list[str]
    defect_support_paths: list[str]
    feature_layer: str
    prototype_family: str = PROTOTYPE_FAMILY_MEAN
    num_prototypes: int | None = None

    @property
    def normal_prototype(self) -> torch.Tensor:
        return self.normal_reference

    @property
    def defect_prototype(self) -> torch.Tensor | None:
        return self.defect_reference


def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    feature_map = F.normalize(feature_map, dim=1)
    return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])


def _normalize_feature_rows(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(features.shape)}")
    return F.normalize(features, dim=1)


def _mean_reference(features: torch.Tensor) -> torch.Tensor:
    prototype = features.mean(dim=0, keepdim=True)
    return F.normalize(prototype, dim=1)


def _kmeans_reference(
    features: torch.Tensor,
    num_prototypes: int,
    seed: int,
    num_iters: int,
) -> torch.Tensor:
    features = _normalize_feature_rows(features)
    num_features = features.shape[0]
    if num_features == 0:
        raise ValueError("Cannot cluster an empty feature bank.")
    if num_prototypes <= 0:
        raise ValueError(f"num_prototypes must be positive, got {num_prototypes}")

    k = min(num_prototypes, num_features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    init_indices = torch.randperm(num_features, generator=generator)[:k]
    centers = features[init_indices].clone()

    for _ in range(max(1, int(num_iters))):
        assignments = (features @ centers.t()).argmax(dim=1)
        updated_centers: list[torch.Tensor] = []
        for index in range(k):
            mask = assignments == index
            if mask.any():
                center = features[mask].mean(dim=0, keepdim=True)
            else:
                fallback_index = int(torch.randint(0, num_features, (1,), generator=generator).item())
                center = features[fallback_index : fallback_index + 1]
            updated_centers.append(F.normalize(center, dim=1))
        centers = torch.cat(updated_centers, dim=0)
    return centers.contiguous()


def build_reference_bank(
    features: torch.Tensor,
    prototype_family: str = PROTOTYPE_FAMILY_MEAN,
    num_prototypes: int = 1,
    seed: int = 42,
    num_iters: int = 25,
) -> torch.Tensor:
    features = _normalize_feature_rows(features)
    if features.shape[0] == 0:
        raise ValueError("Cannot build a reference bank from an empty feature set.")

    if prototype_family == PROTOTYPE_FAMILY_MEAN:
        return _mean_reference(features)
    if prototype_family == PROTOTYPE_FAMILY_MEMORY_BANK:
        return features.contiguous()
    if prototype_family == PROTOTYPE_FAMILY_MULTI_PROTOTYPE:
        return _kmeans_reference(
            features=features,
            num_prototypes=num_prototypes,
            seed=seed,
            num_iters=num_iters,
        )
    raise ValueError(f"Unsupported prototype_family: {prototype_family}")


@torch.no_grad()
def encode_support_set(
    encoder,
    support_samples: Iterable[ImageSample],
    image_size: int,
    device: torch.device,
    feature_layer: str = "local",
) -> tuple[torch.Tensor, list[str]]:
    batch, paths = make_support_batch(list(support_samples), image_size=image_size)
    encoded = encoder(batch.to(device))
    if feature_layer not in encoded:
        raise KeyError(f"Unsupported feature layer: {feature_layer}")
    return flatten_feature_map(encoded[feature_layer]), paths


@torch.no_grad()
def build_prototype_bank(
    encoder,
    support_normal,
    support_defect,
    image_size: int,
    device: torch.device,
    feature_layer: str = "local",
    prototype_family: str = PROTOTYPE_FAMILY_MEAN,
    num_prototypes: int = 1,
    seed: int = 42,
    kmeans_iters: int = 25,
) -> PrototypeBank:
    normal_features, normal_paths = encode_support_set(
        encoder,
        support_normal,
        image_size=image_size,
        device=device,
        feature_layer=feature_layer,
    )
    defect_features = None
    defect_paths: list[str] = []
    if support_defect:
        defect_features, defect_paths = encode_support_set(
            encoder,
            support_defect,
            image_size=image_size,
            device=device,
            feature_layer=feature_layer,
        )

    normal_reference = build_reference_bank(
        features=normal_features,
        prototype_family=prototype_family,
        num_prototypes=num_prototypes,
        seed=seed,
        num_iters=kmeans_iters,
    )
    defect_reference = None
    if defect_features is not None:
        defect_reference = build_reference_bank(
            features=defect_features,
            prototype_family=prototype_family,
            num_prototypes=num_prototypes,
            seed=seed + 1,
            num_iters=kmeans_iters,
        )
    return PrototypeBank(
        normal_reference=normal_reference,
        defect_reference=defect_reference,
        normal_support_paths=normal_paths,
        defect_support_paths=defect_paths,
        feature_layer=feature_layer,
        prototype_family=prototype_family,
        num_prototypes=num_prototypes,
    )
