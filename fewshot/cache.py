import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .data import ImageTransform


@dataclass(frozen=True)
class FeatureCacheSpec:
    cache_root: Path
    image_size: int
    pretrained: str
    feature_layer: str
    seed: int
    cache_version: str = "p0_feature_cache_v2"


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _image_metadata(image_path: Path) -> dict[str, object]:
    stat = image_path.stat()
    return {
        "image_path": str(image_path.resolve()),
        "image_size_bytes": int(stat.st_size),
        "image_mtime_ns": int(stat.st_mtime_ns),
    }


def build_feature_cache_key(
    image_path: str | Path,
    image_size: int,
    pretrained: str | Path | None,
    feature_layer: str,
    seed: int,
    cache_version: str,
) -> str:
    path = _as_path(image_path).resolve()
    payload = {
        **_image_metadata(path),
        "image_size": int(image_size),
        "pretrained": "" if pretrained is None else str(_as_path(pretrained)),
        "feature_layer": str(feature_layer),
        "seed": int(seed),
        "cache_version": str(cache_version),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def feature_cache_path(
    cache_root: str | Path,
    image_path: str | Path,
    image_size: int,
    pretrained: str | Path | None,
    feature_layer: str,
    seed: int,
    cache_version: str,
) -> Path:
    root = _as_path(cache_root)
    key = build_feature_cache_key(
        image_path=image_path,
        image_size=image_size,
        pretrained=pretrained,
        feature_layer=feature_layer,
        seed=seed,
        cache_version=cache_version,
    )
    return root / feature_layer / f"{key}.pt"


def _build_cache_payload(
    feature_map: torch.Tensor,
    image_path: str | Path,
    image_size: int,
    pretrained: str | Path | None,
    feature_layer: str,
    seed: int,
    cache_version: str,
) -> dict[str, object]:
    path = _as_path(image_path).resolve()
    return {
        "metadata": {
            **_image_metadata(path),
            "image_size": int(image_size),
            "pretrained": "" if pretrained is None else str(_as_path(pretrained)),
            "feature_layer": str(feature_layer),
            "seed": int(seed),
            "cache_version": str(cache_version),
        },
        "feature": feature_map.detach().cpu().contiguous(),
    }


def save_feature_cache_entry(
    cache_path: str | Path,
    feature_map: torch.Tensor,
    image_path: str | Path,
    image_size: int,
    pretrained: str | Path | None,
    feature_layer: str,
    seed: int,
    cache_version: str,
) -> Path:
    path = _as_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_cache_payload(
        feature_map=feature_map,
        image_path=image_path,
        image_size=image_size,
        pretrained=pretrained,
        feature_layer=feature_layer,
        seed=seed,
        cache_version=cache_version,
    )
    torch.save(payload, path)
    return path


def load_feature_cache_entry(
    cache_path: str | Path,
    image_path: str | Path,
    image_size: int,
    pretrained: str | Path | None,
    feature_layer: str,
    seed: int,
    cache_version: str,
) -> torch.Tensor:
    path = _as_path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature cache entry: {path}")

    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    expected = _build_cache_payload(
        feature_map=torch.empty(0),
        image_path=image_path,
        image_size=image_size,
        pretrained=pretrained,
        feature_layer=feature_layer,
        seed=seed,
        cache_version=cache_version,
    )["metadata"]
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Feature cache metadata mismatch for {path}: key={key} expected={expected_value} actual={actual_value}"
            )

    feature = payload["feature"]
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise ValueError(f"Cached feature at {path} must be a 4D torch.Tensor.")
    return feature.contiguous()


def _missing_paths(
    image_paths: Sequence[str | Path],
    spec: FeatureCacheSpec,
) -> list[Path]:
    missing: list[Path] = []
    seen: set[str] = set()
    for image_path in image_paths:
        path = _as_path(image_path).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        cache_path = feature_cache_path(
            cache_root=spec.cache_root,
            image_path=path,
            image_size=spec.image_size,
            pretrained=spec.pretrained,
            feature_layer=spec.feature_layer,
            seed=spec.seed,
            cache_version=spec.cache_version,
        )
        if not cache_path.is_file():
            missing.append(path)
            continue
        try:
            load_feature_cache_entry(
                cache_path=cache_path,
                image_path=path,
                image_size=spec.image_size,
                pretrained=spec.pretrained,
                feature_layer=spec.feature_layer,
                seed=spec.seed,
                cache_version=spec.cache_version,
            )
        except Exception:
            missing.append(path)
    return missing


@torch.no_grad()
def populate_feature_cache(
    encoder,
    image_paths: Sequence[str | Path],
    spec: FeatureCacheSpec,
    device: torch.device,
    batch_size: int = 8,
) -> list[Path]:
    missing_paths = _missing_paths(image_paths=image_paths, spec=spec)
    if not missing_paths:
        return []

    transform = ImageTransform(image_size=spec.image_size, augment=False)
    written_paths: list[Path] = []
    for start in range(0, len(missing_paths), batch_size):
        batch_paths = missing_paths[start : start + batch_size]
        batch = torch.stack([transform(path) for path in batch_paths], dim=0).to(device)
        encoded = encoder(batch)
        if spec.feature_layer not in encoded:
            raise ValueError(f"Encoder output does not contain feature layer: {spec.feature_layer}")
        feature_batch = encoded[spec.feature_layer].detach().cpu()
        if feature_batch.ndim != 4:
            raise ValueError(f"Feature layer {spec.feature_layer} must be a 4D map.")
        for index, path in enumerate(batch_paths):
            cache_path = feature_cache_path(
                cache_root=spec.cache_root,
                image_path=path,
                image_size=spec.image_size,
                pretrained=spec.pretrained,
                feature_layer=spec.feature_layer,
                seed=spec.seed,
                cache_version=spec.cache_version,
            )
            save_feature_cache_entry(
                cache_path=cache_path,
                feature_map=feature_batch[index : index + 1],
                image_path=path,
                image_size=spec.image_size,
                pretrained=spec.pretrained,
                feature_layer=spec.feature_layer,
                seed=spec.seed,
                cache_version=spec.cache_version,
            )
            written_paths.append(cache_path)
    return written_paths


def load_feature_cache_batch(
    image_paths: Sequence[str | Path],
    spec: FeatureCacheSpec,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for image_path in image_paths:
        cache_path = feature_cache_path(
            cache_root=spec.cache_root,
            image_path=image_path,
            image_size=spec.image_size,
            pretrained=spec.pretrained,
            feature_layer=spec.feature_layer,
            seed=spec.seed,
            cache_version=spec.cache_version,
        )
        feature = load_feature_cache_entry(
            cache_path=cache_path,
            image_path=image_path,
            image_size=spec.image_size,
            pretrained=spec.pretrained,
            feature_layer=spec.feature_layer,
            seed=spec.seed,
            cache_version=spec.cache_version,
        )
        tensors.append(feature)
    if not tensors:
        raise ValueError("No image paths provided for feature cache batch load.")
    return torch.cat(tensors, dim=0)


def collect_image_paths(samples: Iterable[object]) -> list[Path]:
    paths: list[Path] = []
    for sample in samples:
        path = getattr(sample, "path", sample)
        paths.append(_as_path(path).resolve())
    return paths
