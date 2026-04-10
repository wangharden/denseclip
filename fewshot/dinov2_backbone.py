import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fewshot.data import RESIZE_MODE_SQUARE, resize_pil_image


DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

FEATURE_SOURCE_LAST = "last"
FEATURE_SOURCE_LAST4_MEAN = "last4_mean"
FEATURE_VIEW_BASE = "base"
FEATURE_VIEW_OBJECT_NORMALIZED = "object_normalized"
FEATURE_CACHE_VERSION = "official_mask_v1"
FEATURE_SOURCES = (
    FEATURE_SOURCE_LAST,
    FEATURE_SOURCE_LAST4_MEAN,
)


class DinoImageTransform:
    def __init__(
        self,
        image_size: int,
        resize_mode: str = RESIZE_MODE_SQUARE,
        patch_multiple: int = 14,
    ) -> None:
        self.image_size = int(image_size)
        self.resize_mode = str(resize_mode)
        self.patch_multiple = int(patch_multiple)

    def _load_rgb(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        resized = resize_pil_image(
            image=image,
            image_size=self.image_size,
            resize_mode=self.resize_mode,
            patch_multiple=self.patch_multiple,
            resample=Image.BICUBIC,
        )
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = (rgb - np.asarray(DINO_MEAN, dtype=np.float32)) / np.asarray(DINO_STD, dtype=np.float32)
        return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()

    def _object_normalized_rgb(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        border_pixels = np.concatenate(
            [
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :],
            ],
            axis=0,
        )
        background_color = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(rgb - background_color[None, None, :], axis=2)
        border_distance = np.linalg.norm(border_pixels - background_color[None, :], axis=1)
        threshold = float(border_distance.mean() + 2.5 * border_distance.std())
        threshold = max(threshold, 0.08)
        mask = distance > threshold
        min_foreground = max(16, int(round(mask.size * 0.02)))
        if int(mask.sum()) < min_foreground:
            margin_y = max(2, int(round(height * 0.08)))
            margin_x = max(2, int(round(width * 0.08)))
            top, bottom = margin_y, max(margin_y + 1, height - margin_y)
            left, right = margin_x, max(margin_x + 1, width - margin_x)
        else:
            ys, xs = np.nonzero(mask)
            top = int(ys.min())
            bottom = int(ys.max()) + 1
            left = int(xs.min())
            right = int(xs.max()) + 1
            pad_y = max(2, int(round((bottom - top) * 0.10)))
            pad_x = max(2, int(round((right - left) * 0.10)))
            top = max(0, top - pad_y)
            bottom = min(height, bottom + pad_y)
            left = max(0, left - pad_x)
            right = min(width, right + pad_x)
        cropped = rgb[top:bottom, left:right, :]
        if cropped.size == 0:
            cropped = rgb
        restored = Image.fromarray(np.clip(cropped * 255.0, 0.0, 255.0).astype(np.uint8)).resize(
            (self.image_size, self.image_size),
            resample=Image.BICUBIC,
        )
        return np.asarray(restored, dtype=np.float32) / 255.0

    def __call__(self, path: Path) -> dict[str, torch.Tensor]:
        rgb = self._load_rgb(path)
        return {"image": self._to_tensor(rgb)}


class DinoPathDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        image_size: int,
        resize_mode: str = RESIZE_MODE_SQUARE,
        patch_multiple: int = 14,
    ) -> None:
        self.paths = list(paths)
        self.transform = DinoImageTransform(
            image_size=image_size,
            resize_mode=resize_mode,
            patch_multiple=patch_multiple,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        transformed = self.transform(Path(path))
        return {"path": path, "image": transformed["image"]}


def cache_key_for_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return f"{Path(path).stem}_{digest}.pt"


def compute_foreground_mask(patch_map: torch.Tensor, threshold: float = 10.0, border_fraction: float = 0.2) -> torch.Tensor:
    if patch_map.ndim != 3:
        raise ValueError(f"Expected a 3D patch map, got shape {tuple(patch_map.shape)}")
    channels, height, width = patch_map.shape
    features = patch_map.permute(1, 2, 0).reshape(-1, channels).detach().cpu().numpy().astype(np.float32)
    if features.size == 0:
        return torch.ones((1, height, width), dtype=torch.bool)

    first_pc = PCA(n_components=1, svd_solver="randomized").fit_transform(features).reshape(height, width)
    mask = first_pc > float(threshold)
    inner_top = int(height * border_fraction)
    inner_bottom = int(height * (1.0 - border_fraction))
    inner_left = int(width * border_fraction)
    inner_right = int(width * (1.0 - border_fraction))
    center = mask[inner_top:inner_bottom, inner_left:inner_right]
    if center.size > 0 and float(center.mean()) <= 0.35:
        mask = (-first_pc) > float(threshold)

    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_dilation(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    if not mask.any():
        mask = np.ones((height, width), dtype=bool)
    return torch.from_numpy(mask[None, ...].astype(np.bool_))


def apply_foreground_mask(patch_map: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    if foreground_mask.ndim == 2:
        foreground_mask = foreground_mask.unsqueeze(0)
    if foreground_mask.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D foreground mask, got shape {tuple(foreground_mask.shape)}")
    return patch_map * foreground_mask.to(dtype=patch_map.dtype, device=patch_map.device)


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
    resize_mode: str,
    patch_multiple: int,
    feature_source: str,
    batch_size: int,
    workers: int,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_paths: list[str] = []
    for path in image_paths:
        cache_path = cache_dir / cache_key_for_path(path)
        if not cache_path.is_file():
            missing_paths.append(path)
            continue
        payload = torch.load(cache_path, map_location="cpu")
        if (
            payload.get("cache_version") != FEATURE_CACHE_VERSION
            or
            "patch_map" not in payload
            or "cls_token" not in payload
            or "object_normalized_patch_map" not in payload
            or "object_normalized_cls_token" not in payload
            or payload.get("feature_source") != feature_source
            or int(payload.get("image_size", -1)) != int(image_size)
            or str(payload.get("resize_mode", RESIZE_MODE_SQUARE)) != str(resize_mode)
            or int(payload.get("patch_multiple", 14)) != int(patch_multiple)
        ):
            missing_paths.append(path)
    if not missing_paths:
        return []

    dataset = DinoPathDataset(
        paths=missing_paths,
        image_size=image_size,
        resize_mode=resize_mode,
        patch_multiple=patch_multiple,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    written: list[str] = []
    for batch in loader:
        patch_maps, cls_tokens = encoder.encode_batch(batch["image"], feature_source=feature_source)
        for path, patch_map, cls_token in zip(
            batch["path"],
            patch_maps,
            cls_tokens,
            strict=True,
        ):
            foreground_mask = compute_foreground_mask(patch_map)
            normalized_patch_map = apply_foreground_mask(patch_map, foreground_mask)
            torch.save(
                {
                    "path": str(path),
                    "cache_version": FEATURE_CACHE_VERSION,
                    "feature_source": feature_source,
                    "image_size": int(image_size),
                    "resize_mode": str(resize_mode),
                    "patch_multiple": int(patch_multiple),
                    "patch_map": patch_map.detach().cpu().half().contiguous(),
                    "cls_token": cls_token.detach().cpu().half().contiguous(),
                    "object_normalized_patch_map": normalized_patch_map.detach().cpu().half().contiguous(),
                    "object_normalized_cls_token": cls_token.detach().cpu().half().contiguous(),
                },
                cache_dir / cache_key_for_path(path),
            )
            written.append(str(path))
    return written


def load_dinov2_feature_cache_batch(
    image_paths: list[str | Path],
    cache_dir: Path,
    device: torch.device | None = None,
    view: str = FEATURE_VIEW_BASE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not image_paths:
        empty_patch = torch.empty((0, 0, 0, 0), dtype=torch.float32)
        empty_cls = torch.empty((0, 0), dtype=torch.float32)
        if device is not None:
            empty_patch = empty_patch.to(device)
            empty_cls = empty_cls.to(device)
        return empty_patch, empty_cls
    patch_maps: list[torch.Tensor] = []
    cls_tokens: list[torch.Tensor] = []
    if view == FEATURE_VIEW_BASE:
        patch_key = "patch_map"
        cls_key = "cls_token"
    elif view == FEATURE_VIEW_OBJECT_NORMALIZED:
        patch_key = "object_normalized_patch_map"
        cls_key = "object_normalized_cls_token"
    else:
        raise ValueError(f"Unsupported DINO cache view: {view}")
    for image_path in image_paths:
        payload = torch.load(cache_dir / cache_key_for_path(image_path), map_location="cpu")
        patch_maps.append(payload[patch_key].float())
        cls_tokens.append(payload[cls_key].float())
    patch_batch = torch.stack(patch_maps, dim=0)
    cls_batch = torch.stack(cls_tokens, dim=0)
    if device is not None:
        patch_batch = patch_batch.to(device)
        cls_batch = cls_batch.to(device)
    return patch_batch, cls_batch


def flatten_patch_map(patch_map: torch.Tensor) -> torch.Tensor:
    if patch_map.numel() == 0:
        return torch.empty((0, 0), dtype=patch_map.dtype, device=patch_map.device)
    patch_map = F.normalize(patch_map, dim=1, eps=1e-6)
    flat = patch_map.permute(0, 2, 3, 1).reshape(-1, patch_map.shape[1])
    valid = flat.norm(dim=1) > 1e-6
    if not bool(valid.any()):
        return flat[:0]
    return flat[valid]


__all__ = [
    "DinoV2PatchEncoder",
    "FEATURE_SOURCE_LAST",
    "FEATURE_SOURCE_LAST4_MEAN",
    "FEATURE_SOURCES",
    "FEATURE_VIEW_BASE",
    "FEATURE_VIEW_OBJECT_NORMALIZED",
    "FEATURE_CACHE_VERSION",
    "apply_foreground_mask",
    "cache_key_for_path",
    "compute_foreground_mask",
    "flatten_patch_map",
    "load_dinov2_feature_cache_batch",
    "populate_dinov2_feature_cache",
]
