import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_SUPPORT_TENSOR_CACHE: dict[tuple[str, int], torch.Tensor] = {}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int | None
    category: str
    stem: str
    defect_type: str | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class StageA1Split:
    support_normal: list[ImageSample]
    test_query: list[ImageSample]


@dataclass(frozen=True)
class StageBSplit:
    support_normal: list[ImageSample]
    support_defect: list[ImageSample]
    test_query: list[ImageSample]


@dataclass(frozen=True)
class SharedSplitManifest:
    category: str
    support_normal: list[ImageSample]
    support_defect: list[ImageSample]
    query_eval: list[ImageSample]
    metadata: dict[str, object] | None = None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def _normalize_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
    rgb_array = rgb_array.astype(np.float32) / 255.0
    rgb_array = (rgb_array - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    return rgb_array


class ImageTransform:
    def __init__(self, image_size: int, augment: bool = False) -> None:
        self.image_size = image_size
        self.augment = augment

    def __call__(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        resized = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        rgb_array = _normalize_rgb_array(np.asarray(resized, dtype=np.float32))
        return torch.from_numpy(rgb_array.transpose(2, 0, 1))


def load_image_rgb(path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    resized = image.resize((image_size, image_size), resample=Image.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def load_mask_array(mask_path: str | Path | None, image_size: int) -> np.ndarray:
    if mask_path is None:
        return np.zeros((image_size, image_size), dtype=np.float32)
    mask = Image.open(mask_path).convert("L")
    resized = mask.resize((image_size, image_size), resample=Image.NEAREST)
    mask_array = np.asarray(resized, dtype=np.float32)
    return (mask_array > 0).astype(np.float32)


def load_support_tensor_cached(path: str | Path, image_size: int) -> torch.Tensor:
    resolved_path = Path(path).resolve()
    cache_key = (str(resolved_path), int(image_size))
    cached = _SUPPORT_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    transform = ImageTransform(image_size=image_size, augment=False)
    tensor = transform(resolved_path).contiguous()
    _SUPPORT_TENSOR_CACHE[cache_key] = tensor
    return tensor


def _sorted_images(root: Path) -> list[Path]:
    return [path for path in sorted(root.iterdir()) if _is_image_file(path)]


def _serialize_sample(sample: ImageSample) -> dict[str, object]:
    return {
        "path": str(sample.path),
        "label": sample.label,
        "category": sample.category,
        "stem": sample.stem,
        "defect_type": sample.defect_type,
        "mask_path": None if sample.mask_path is None else str(sample.mask_path),
    }


def _deserialize_sample(payload: dict[str, object]) -> ImageSample:
    return ImageSample(
        path=Path(str(payload["path"])),
        label=None if payload["label"] is None else int(payload["label"]),
        category=str(payload["category"]),
        stem=str(payload["stem"]),
        defect_type=None if payload["defect_type"] is None else str(payload["defect_type"]),
        mask_path=None if payload["mask_path"] is None else Path(str(payload["mask_path"])),
    )


def _collect_test_samples(category_root: Path, category: str) -> list[ImageSample]:
    test_root = category_root / "test"
    gt_root = category_root / "ground_truth"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    test_query: list[ImageSample] = []
    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        for image_path in _sorted_images(defect_dir):
            if defect_type == "good":
                mask_path = None
                label = 0
            else:
                mask_path = gt_root / defect_type / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path}: expected {mask_path}")
                label = 1
            test_query.append(
                ImageSample(
                    path=image_path,
                    label=label,
                    category=category,
                    stem=image_path.stem,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    if not test_query:
        raise ValueError(f"No test images found in {test_root}")
    return test_query


def build_shared_split_manifest(
    root: str | Path,
    category: str,
    support_normal_k: int,
    support_defect_k: int = 0,
    seed: int = 42,
) -> SharedSplitManifest:
    category_root = Path(root) / category
    train_good_dir = category_root / "train" / "good"
    if not train_good_dir.is_dir():
        raise FileNotFoundError(f"Missing support directory: {train_good_dir}")

    normal_train_paths = _sorted_images(train_good_dir)
    if len(normal_train_paths) < support_normal_k:
        raise ValueError(
            f"Not enough normal support images for {category}: "
            f"requested {support_normal_k}, found {len(normal_train_paths)}"
        )

    rng = random.Random(seed)
    shuffled_normal = list(normal_train_paths)
    rng.shuffle(shuffled_normal)
    support_normal_paths = sorted(shuffled_normal[:support_normal_k])
    support_normal = [
        ImageSample(path=path, label=0, category=category, stem=path.stem, defect_type="good", mask_path=None)
        for path in support_normal_paths
    ]

    test_samples = _collect_test_samples(category_root=category_root, category=category)
    support_defect: list[ImageSample] = []
    if support_defect_k > 0:
        defect_samples = [sample for sample in test_samples if sample.label == 1]
        if len(defect_samples) < support_defect_k:
            raise ValueError(
                f"Not enough defect support images for {category}: "
                f"requested {support_defect_k}, found {len(defect_samples)}"
            )
        shuffled_defect = list(defect_samples)
        rng.shuffle(shuffled_defect)
        support_defect = sorted(shuffled_defect[:support_defect_k], key=lambda sample: str(sample.path))

    support_defect_keys = {str(sample.path) for sample in support_defect}
    query_eval = [sample for sample in test_samples if str(sample.path) not in support_defect_keys]
    metadata = {
        "seed": seed,
        "requested_support_normal_k": support_normal_k,
        "requested_support_defect_k": support_defect_k,
        "num_support_normal": len(support_normal),
        "num_support_defect": len(support_defect),
        "num_query_eval": len(query_eval),
    }
    return SharedSplitManifest(
        category=category,
        support_normal=support_normal,
        support_defect=support_defect,
        query_eval=query_eval,
        metadata=metadata,
    )


def save_shared_split_manifest(
    manifest: SharedSplitManifest,
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata = dict(manifest.metadata or {})
    if metadata:
        merged_metadata.update(metadata)
    payload: dict[str, object] = {
        "version": 1,
        "category": manifest.category,
        "support_normal": [_serialize_sample(sample) for sample in manifest.support_normal],
        "support_defect": [_serialize_sample(sample) for sample in manifest.support_defect],
        "query_eval": [_serialize_sample(sample) for sample in manifest.query_eval],
    }
    if merged_metadata:
        payload["metadata"] = merged_metadata
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_shared_split_manifest(path: str | Path) -> SharedSplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SharedSplitManifest(
        category=str(payload["category"]),
        support_normal=[_deserialize_sample(item) for item in payload["support_normal"]],
        support_defect=[_deserialize_sample(item) for item in payload.get("support_defect", [])],
        query_eval=[_deserialize_sample(item) for item in payload["query_eval"]],
        metadata=payload.get("metadata"),
    )


def stage_a1_split_from_manifest(manifest: SharedSplitManifest) -> StageA1Split:
    return StageA1Split(
        support_normal=list(manifest.support_normal),
        test_query=list(manifest.query_eval),
    )


def stage_b_split_from_manifest(manifest: SharedSplitManifest) -> StageBSplit:
    return StageBSplit(
        support_normal=list(manifest.support_normal),
        support_defect=list(manifest.support_defect),
        test_query=list(manifest.query_eval),
    )


def build_stage_a1_split(
    root: str | Path,
    category: str,
    support_normal_k: int,
    seed: int = 42,
) -> StageA1Split:
    manifest = build_shared_split_manifest(
        root=root,
        category=category,
        support_normal_k=support_normal_k,
        support_defect_k=0,
        seed=seed,
    )
    return stage_a1_split_from_manifest(manifest)


def build_stage_b_split(
    root: str | Path,
    category: str,
    support_normal_k: int,
    support_defect_k: int,
    seed: int = 42,
) -> StageBSplit:
    manifest = build_shared_split_manifest(
        root=root,
        category=category,
        support_normal_k=support_normal_k,
        support_defect_k=support_defect_k,
        seed=seed,
    )
    return stage_b_split_from_manifest(manifest)


class StageA1Dataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], image_size: int, augment: bool = False) -> None:
        self.samples = list(samples)
        self.transform = ImageTransform(image_size=image_size, augment=augment)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        mask = load_mask_array(sample.mask_path, image_size=self.image_size)
        return {
            "image": self.transform(sample.path),
            "label": int(sample.label or 0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "path": str(sample.path),
            "stem": sample.stem,
            "category": sample.category,
            "defect_type": sample.defect_type or "unknown",
            "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
        }


def make_support_batch(samples: Sequence[ImageSample], image_size: int) -> tuple[torch.Tensor, list[str]]:
    tensors = []
    paths = []
    for sample in samples:
        tensors.append(load_support_tensor_cached(sample.path, image_size=image_size))
        paths.append(str(sample.path))
    if not tensors:
        raise ValueError("Support set is empty.")
    return torch.stack(tensors, dim=0), paths
