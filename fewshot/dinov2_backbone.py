import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

FEATURE_SOURCE_LAST = "last"
FEATURE_SOURCE_LAST4_MEAN = "last4_mean"
FEATURE_SOURCES = (
    FEATURE_SOURCE_LAST,
    FEATURE_SOURCE_LAST4_MEAN,
)


class DinoImageTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        resized = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        rgb = np.asarray(resized, dtype=np.float32) / 255.0
        rgb = (rgb - np.asarray(DINO_MEAN, dtype=np.float32)) / np.asarray(DINO_STD, dtype=np.float32)
        return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()


class DinoPathDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = list(paths)
        self.transform = DinoImageTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        return {"path": path, "image": self.transform(Path(path))}


def cache_key_for_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return f"{Path(path).stem}_{digest}.pt"


class DinoV2PatchEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = str(model_name)
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name).to(device).eval()

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor, feature_source: str) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        if feature_source == FEATURE_SOURCE_LAST:
            outputs = self.model.forward_features(images)
            patch_tokens = outputs["x_norm_patchtokens"]
            cls_tokens = outputs["x_norm_clstoken"]
            side = int(round(patch_tokens.shape[1] ** 0.5))
            patch_maps = patch_tokens.reshape(images.shape[0], side, side, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
            return patch_maps.float(), cls_tokens.float()
        if feature_source == FEATURE_SOURCE_LAST4_MEAN:
            layers = self.model.get_intermediate_layers(images, n=4, reshape=True, return_class_token=True)
            patch_maps = torch.stack([item[0].float() for item in layers], dim=0).mean(dim=0)
            cls_tokens = torch.stack([item[1].float() for item in layers], dim=0).mean(dim=0)
            return patch_maps, cls_tokens
        raise ValueError(f"Unsupported DINO feature source: {feature_source}")


@torch.no_grad()
def populate_dinov2_feature_cache(
    encoder: DinoV2PatchEncoder,
    image_paths: list[str],
    cache_dir: Path,
    image_size: int,
    feature_source: str,
    batch_size: int,
    workers: int,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_paths = [path for path in image_paths if not (cache_dir / cache_key_for_path(path)).is_file()]
    if not missing_paths:
        return []

    dataset = DinoPathDataset(paths=missing_paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    written: list[str] = []
    for batch in loader:
        patch_maps, cls_tokens = encoder.encode_batch(batch["image"], feature_source=feature_source)
        for path, patch_map, cls_token in zip(batch["path"], patch_maps, cls_tokens, strict=True):
            torch.save(
                {
                    "path": str(path),
                    "feature_source": feature_source,
                    "image_size": int(image_size),
                    "patch_map": patch_map.detach().cpu().half().contiguous(),
                    "cls_token": cls_token.detach().cpu().half().contiguous(),
                },
                cache_dir / cache_key_for_path(path),
            )
            written.append(str(path))
    return written


def load_dinov2_feature_cache_batch(
    image_paths: list[str | Path],
    cache_dir: Path,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_maps: list[torch.Tensor] = []
    cls_tokens: list[torch.Tensor] = []
    for image_path in image_paths:
        payload = torch.load(cache_dir / cache_key_for_path(image_path), map_location="cpu")
        patch_maps.append(payload["patch_map"].float())
        cls_tokens.append(payload["cls_token"].float())
    patch_batch = torch.stack(patch_maps, dim=0)
    cls_batch = torch.stack(cls_tokens, dim=0)
    if device is not None:
        patch_batch = patch_batch.to(device)
        cls_batch = cls_batch.to(device)
    return patch_batch, cls_batch


def flatten_patch_map(patch_map: torch.Tensor) -> torch.Tensor:
    patch_map = F.normalize(patch_map, dim=1, eps=1e-6)
    return patch_map.permute(0, 2, 3, 1).reshape(-1, patch_map.shape[1])


__all__ = [
    "DinoV2PatchEncoder",
    "FEATURE_SOURCE_LAST",
    "FEATURE_SOURCE_LAST4_MEAN",
    "FEATURE_SOURCES",
    "cache_key_for_path",
    "flatten_patch_map",
    "load_dinov2_feature_cache_batch",
    "populate_dinov2_feature_cache",
]
