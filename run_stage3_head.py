import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.data import load_mask_array
from fewshot.head import ImagePoolHead, MapFusionHead
from fewshot.losses import dice_loss, pairwise_margin_rank
from fewshot.stage_a1 import binary_auroc, pixel_auroc


MODE_P1 = "p1"
MODE_P2 = "p2"
MAP_KEYS = (
    "base_anomaly_map",
    "subspace_residual_map",
    "knn_top1_map",
    "knn_top3_map",
    "knn_gap_map",
)
PERCENTILES = (75.0, 90.0, 95.0, 99.0)
IDENTITY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage3 P1/P2 heads from cached Stage2 winner maps.")
    parser.add_argument("--mode", default=MODE_P1, choices=(MODE_P1, MODE_P2))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--map-hidden-channels", type=int, default=32)
    parser.add_argument("--image-hidden-channels", type=int, default=64)
    parser.add_argument("--map-topk-ratio", type=float, default=0.1)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--img-bce-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--mask-pos-weight-max", type=float, default=20.0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    def cuda_is_usable() -> bool:
        if not torch.cuda.is_available():
            return False
        arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        if not arch_list:
            return True
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}" in arch_list

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if not cuda_is_usable():
            raise RuntimeError(
                "Requested --device cuda, but the current PyTorch CUDA build does not support this GPU architecture."
            )
        return torch.device("cuda")
    if cuda_is_usable():
        return torch.device("cuda")
    if torch.cuda.is_available():
        print(json.dumps({"event": "device_fallback", "requested": "auto", "selected": "cpu", "reason": "cuda_arch_unsupported"}))
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    labeled, num = ndimage.label(mask > 0.5)
    return [(labeled == label_id) for label_id in range(1, num + 1)]


def pro_score(
    mask_arrays: list[np.ndarray],
    score_maps: list[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    if not mask_arrays or not score_maps:
        return float("nan")
    all_scores = np.concatenate([score.reshape(-1) for score in score_maps], axis=0)
    negative_scores = np.concatenate(
        [score[mask <= 0.5].reshape(-1) for mask, score in zip(mask_arrays, score_maps)],
        axis=0,
    )
    regions = [connected_component_masks(mask) for mask in mask_arrays]
    if negative_scores.size == 0 or sum(len(items) for items in regions) == 0:
        return float("nan")

    thresholds = np.linspace(float(all_scores.max()), float(all_scores.min()), num_thresholds)
    fprs: list[float] = []
    pros: list[float] = []
    for threshold in thresholds:
        fpr = float((negative_scores >= threshold).mean())
        if fpr > max_fpr:
            continue
        overlaps: list[float] = []
        for region_set, score_map in zip(regions, score_maps):
            pred = score_map >= threshold
            for region_mask in region_set:
                overlaps.append(float(pred[region_mask].mean()))
        if overlaps:
            fprs.append(fpr)
            pros.append(float(np.mean(overlaps)))
    if len(fprs) < 2:
        return float("nan")
    order = np.argsort(fprs)
    return float(np.trapezoid(np.asarray(pros)[order], np.asarray(fprs)[order] / max_fpr))


def summarize_map(score_map: torch.Tensor) -> list[float]:
    flat = score_map.reshape(-1).float()
    stats = [
        float(flat.max().item()),
        float(flat.mean().item()),
        float(flat.topk(k=max(1, int(flat.numel() * 0.1))).values.mean().item()),
        float(flat.std(unbiased=False).item()),
    ]
    for percentile in PERCENTILES:
        stats.append(float(torch.quantile(flat, percentile / 100.0).item()))
    return stats


def build_feature_vector(payload: dict[str, object]) -> list[float]:
    features: list[float] = []
    for key in MAP_KEYS:
        features.extend(summarize_map(payload[key]))
    features.append(float(payload["image_score_base"]))
    return features


def upsample_patch_map(score_map: torch.Tensor, image_size: int) -> np.ndarray:
    return (
        F.interpolate(score_map.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def load_records(cache_dir: Path) -> tuple[list[dict[str, object]], int]:
    cache_config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    image_size = int(cache_config["image_size"])
    index_rows = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for row in index_rows:
        payload = torch.load(row["cache_path"], map_location="cpu")
        map_tensor = torch.cat([payload[key].float() for key in MAP_KEYS], dim=0)
        frozen_patch_map = payload["subspace_residual_map"].float()
        mask_tensor = payload["mask"].float() if "mask" in payload else torch.empty(0, dtype=torch.float32)
        records.append(
            {
                **row,
                "feature_vector": build_feature_vector(payload),
                "map_tensor": map_tensor,
                "frozen_patch_map": frozen_patch_map,
                "pixel_map": upsample_patch_map(frozen_patch_map, image_size=image_size),
                "mask_tensor": mask_tensor,
            }
        )
    return records, image_size


def split_records(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_records = [record for record in records if record["role"] in ("support_normal", "support_defect")]
    eval_records = [record for record in records if record["role"] == "query_eval"]
    if not train_records or not eval_records:
        raise ValueError("Cache must include both train and query_eval records.")
    labels = {int(record["label"]) for record in train_records}
    if labels != {0, 1}:
        raise ValueError(f"Train cache must include both support_normal and support_defect labels, got {sorted(labels)}")
    return train_records, eval_records


def normalize_features(
    train_records: list[dict[str, object]],
    eval_records: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    train_x = torch.tensor([record["feature_vector"] for record in train_records], dtype=torch.float32)
    eval_x = torch.tensor([record["feature_vector"] for record in eval_records], dtype=torch.float32)
    train_y = torch.tensor([float(record["label"]) for record in train_records], dtype=torch.float32)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized_train = (train_x - mean) / std
    normalized_eval = (eval_x - mean) / std
    return normalized_train, train_y, normalized_eval, {"mean": mean.tolist(), "std": std.tolist()}


def normalize_map_tensors(
    train_records: list[dict[str, object]],
    eval_records: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    train_maps = torch.stack([record["map_tensor"] for record in train_records], dim=0)
    eval_maps = torch.stack([record["map_tensor"] for record in eval_records], dim=0)
    mean = train_maps.mean(dim=(0, 2, 3), keepdim=True)
    std = train_maps.std(dim=(0, 2, 3), unbiased=False, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized_train = (train_maps - mean) / std
    normalized_eval = (eval_maps - mean) / std
    stats = {
        "mean": mean.squeeze(0).squeeze(-1).squeeze(-1).tolist(),
        "std": std.squeeze(0).squeeze(-1).squeeze(-1).tolist(),
    }
    return normalized_train, normalized_eval, stats


def fit_score_calibration(
    positive_values: torch.Tensor,
    negative_values: torch.Tensor,
) -> dict[str, float]:
    positive_values = positive_values.reshape(-1).float()
    negative_values = negative_values.reshape(-1).float()
    combined = torch.cat([negative_values, positive_values], dim=0)
    if combined.numel() == 0:
        return {
            "direction": 1.0,
            "center": 0.0,
            "scale": 1.0,
            "positive_mean": 0.0,
            "negative_mean": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }

    if positive_values.numel() == 0 or negative_values.numel() == 0:
        center = float(combined.mean().item())
        scale = float(combined.std(unbiased=False).item())
        return {
            "direction": 1.0,
            "center": center,
            "scale": max(scale, 1e-6),
            "positive_mean": float(positive_values.mean().item()) if positive_values.numel() else center,
            "negative_mean": float(negative_values.mean().item()) if negative_values.numel() else center,
            "positive_count": float(positive_values.numel()),
            "negative_count": float(negative_values.numel()),
        }

    raw_positive_mean = float(positive_values.mean().item())
    raw_negative_mean = float(negative_values.mean().item())
    direction = 1.0 if raw_positive_mean >= raw_negative_mean else -1.0
    oriented_positive = positive_values * direction
    oriented_negative = negative_values * direction
    positive_mean = float(oriented_positive.mean().item())
    negative_mean = float(oriented_negative.mean().item())
    center = 0.5 * (positive_mean + negative_mean)
    pooled = torch.cat([oriented_negative, oriented_positive], dim=0)
    pooled_std = float(pooled.std(unbiased=False).item()) if pooled.numel() > 1 else 0.0
    scale = max((positive_mean - negative_mean) / 2.0, pooled_std, 1e-6)
    return {
        "direction": direction,
        "center": center,
        "scale": scale,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_count": float(positive_values.numel()),
        "negative_count": float(negative_values.numel()),
    }


def fit_image_score_calibration(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    positive_values = scores[labels > 0.5]
    negative_values = scores[labels <= 0.5]
    return fit_score_calibration(positive_values=positive_values, negative_values=negative_values)


def fit_pixel_score_calibration(score_maps: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    positive_values = score_maps[masks > 0.5]
    negative_values = score_maps[masks <= 0.5]
    return fit_score_calibration(positive_values=positive_values, negative_values=negative_values)


def apply_score_calibration(scores: torch.Tensor, calibration: dict[str, float]) -> torch.Tensor:
    direction = float(calibration["direction"])
    center = float(calibration["center"])
    scale = float(calibration["scale"])
    return (scores * direction - center) / scale


def append_jsonl(path: Path | None, row: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def downsample_mask(mask_tensor: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    if mask_tensor.numel() == 0:
        return torch.zeros((1, output_hw[0], output_hw[1]), dtype=torch.float32)
    if tuple(mask_tensor.shape[-2:]) == output_hw:
        return mask_tensor.float()
    return F.interpolate(mask_tensor.unsqueeze(0), size=output_hw, mode="nearest")[0].float()


def build_dual_head_train_targets(
    train_records: list[dict[str, object]],
    output_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor([float(record["label"]) for record in train_records], dtype=torch.float32)
    masks = torch.stack([downsample_mask(record["mask_tensor"], output_hw) for record in train_records], dim=0)
    consistency_targets = torch.stack([record["frozen_patch_map"] for record in train_records], dim=0)
    consistency_selector = torch.tensor(
        [int(record["label"]) == 0 or str(record["category"]) == "bottle" for record in train_records],
        dtype=torch.bool,
    )
    baseline_scores = torch.tensor(
        [float(record["winner_image_score"]) for record in train_records],
        dtype=torch.float32,
    )
    return labels, masks, consistency_targets, consistency_selector, baseline_scores


def evaluate_per_category(
    eval_records: list[dict[str, object]],
    image_scores: list[float],
    pixel_scores: list[np.ndarray],
    image_size: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, object], float, np.ndarray]]] = {}
    for record, score, pixel_map in zip(eval_records, image_scores, pixel_scores):
        grouped.setdefault(str(record["category"]), []).append((record, score, pixel_map))

    rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        image_labels = [int(record["label"]) for record, _, _ in items]
        category_scores = [float(score) for _, score, _ in items]
        category_pixel_scores = [pixel_map for _, _, pixel_map in items]
        pixel_masks = [load_mask_array(record["mask_path"] or None, image_size=image_size) for record, _, _ in items]
        image_auroc = binary_auroc(image_labels, category_scores)
        pixel_auroc_value = pixel_auroc(pixel_masks, category_pixel_scores)
        pro_value = pro_score(pixel_masks, category_pixel_scores)
        rows.append(
            {
                "category": category,
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc_value,
                "pro": pro_value,
                "balanced": (image_auroc + pixel_auroc_value) / 2.0,
            }
        )
    return rows


def aggregate_rows(
    rows: list[dict[str, object]],
    experiment: str,
    pixel_source: str,
    balanced_mode: str,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "num_categories": len(rows),
        "image_auroc_mean": float(np.mean([row["image_auroc"] for row in rows])),
        "pixel_auroc_mean": float(np.mean([row["pixel_auroc"] for row in rows])),
        "pro_mean": float(np.mean([row["pro"] for row in rows])),
        "balanced_mean": float(np.mean([row["balanced"] for row in rows])),
        "pixel_source": pixel_source,
        "balanced_mode": balanced_mode,
    }


def _binary_pos_weight(labels: torch.Tensor) -> float:
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives <= 0.0:
        return 1.0
    return max(1.0, negatives / positives)


def initialize_dual_head_as_identity(model: MapFusionHead) -> None:
    with torch.no_grad():
        model.pixel_head.weight.zero_()
        if model.pixel_head.bias is not None:
            model.pixel_head.bias.zero_()
        final_linear = model.image_head[-1]
        final_linear.weight.zero_()
        if final_linear.bias is not None:
            final_linear.bias.zero_()


def train_image_head(
    model: ImagePoolHead,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    metrics_path: Path | None = None,
) -> list[dict[str, float]]:
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = _binary_pos_weight(train_y)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_rank = 0.0
        steps = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            bce = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
            rank = pairwise_margin_rank(logits, batch_y, margin=args.margin)
            loss = bce + rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_bce += float(bce.item())
            epoch_rank += float(rank.item())
            steps += 1
        row = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "bce": epoch_bce / max(1, steps),
            "rank": epoch_rank / max(1, steps),
            "pos_weight": pos_weight_value,
        }
        history.append(row)
        append_jsonl(metrics_path, {"mode": MODE_P1, **row})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))
    return history


def train_dual_head(
    model: MapFusionHead,
    train_maps: torch.Tensor,
    train_labels: torch.Tensor,
    train_masks: torch.Tensor,
    consistency_targets: torch.Tensor,
    consistency_selector: torch.Tensor,
    baseline_scores: torch.Tensor,
    image_calibration: dict[str, float],
    pixel_calibration: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    metrics_path: Path | None = None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    loader = DataLoader(
        TensorDataset(train_maps, train_labels, train_masks, consistency_targets, consistency_selector, baseline_scores),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    img_pos_weight_value = _binary_pos_weight(train_labels)
    img_pos_weight = torch.tensor([img_pos_weight_value], dtype=torch.float32, device=device)
    mask_positives = float(train_masks.sum().item())
    mask_negatives = float(train_masks.numel() - mask_positives)
    if mask_positives <= 0.0:
        mask_pos_weight_value = 1.0
    else:
        mask_pos_weight_value = min(args.mask_pos_weight_max, max(1.0, mask_negatives / mask_positives))
    mask_pos_weight = torch.tensor([mask_pos_weight_value], dtype=torch.float32, device=device)

    best_loss = float("inf")
    stale_epochs = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "mask_bce": 0.0,
            "dice": 0.0,
            "img_bce": 0.0,
            "rank": 0.0,
            "consistency": 0.0,
        }
        steps = 0
        for batch_maps, batch_labels, batch_masks, batch_targets, batch_selector, batch_baseline_scores in loader:
            batch_maps = batch_maps.to(device)
            batch_labels = batch_labels.to(device)
            batch_masks = batch_masks.to(device)
            batch_targets = batch_targets.to(device)
            batch_selector = batch_selector.to(device)
            batch_baseline_scores = batch_baseline_scores.to(device)

            pixel_residual, image_residual = model(batch_maps)
            final_pixel_scores = pixel_residual + batch_targets
            final_image_scores = image_residual + batch_baseline_scores
            pixel_logits = apply_score_calibration(final_pixel_scores, pixel_calibration)
            image_logits = apply_score_calibration(final_image_scores, image_calibration)

            mask_bce = F.binary_cross_entropy_with_logits(pixel_logits, batch_masks, pos_weight=mask_pos_weight)
            dice = dice_loss(pixel_logits, batch_masks)
            img_bce = F.binary_cross_entropy_with_logits(image_logits, batch_labels, pos_weight=img_pos_weight)
            rank = pairwise_margin_rank(image_logits, batch_labels, margin=args.margin, smooth=True)
            if bool(batch_selector.any().item()):
                consistency = F.mse_loss(final_pixel_scores[batch_selector], batch_targets[batch_selector])
            else:
                consistency = final_pixel_scores.new_zeros(())

            loss = (
                args.mask_bce_weight * mask_bce
                + args.dice_weight * dice
                + args.img_bce_weight * img_bce
                + args.rank_weight * rank
                + args.consistency_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sums["loss"] += float(loss.item())
            sums["mask_bce"] += float(mask_bce.item())
            sums["dice"] += float(dice.item())
            sums["img_bce"] += float(img_bce.item())
            sums["rank"] += float(rank.item())
            sums["consistency"] += float(consistency.item())
            steps += 1

        row = {"epoch": epoch, **{key: value / max(1, steps) for key, value in sums.items()}}
        row["img_pos_weight"] = img_pos_weight_value
        row["mask_pos_weight"] = mask_pos_weight_value
        history.append(row)
        append_jsonl(metrics_path, {"mode": MODE_P2, **row})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))

        if row["loss"] < best_loss - args.min_delta:
            best_loss = row["loss"]
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(json.dumps({"event": "early_stop", "epoch": epoch, "best_loss": best_loss}))
                break
    return history, best_state


@torch.no_grad()
def predict_image_scores(model: ImagePoolHead, features: torch.Tensor, device: torch.device) -> list[float]:
    model.eval()
    logits = model(features.to(device))
    return torch.sigmoid(logits).cpu().tolist()


@torch.no_grad()
def predict_dual_head(
    model: MapFusionHead,
    eval_maps: torch.Tensor,
    eval_frozen_patch_maps: torch.Tensor,
    eval_baseline_scores: torch.Tensor,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[np.ndarray]]:
    model.eval()
    loader = DataLoader(
        TensorDataset(eval_maps, eval_frozen_patch_maps, eval_baseline_scores),
        batch_size=batch_size,
        shuffle=False,
    )
    image_scores: list[float] = []
    pixel_scores: list[np.ndarray] = []
    for batch_maps, batch_frozen_patch, batch_baseline_scores in loader:
        batch_maps = batch_maps.to(device)
        batch_frozen_patch = batch_frozen_patch.to(device)
        batch_baseline_scores = batch_baseline_scores.to(device)
        pixel_residual, image_residual = model(batch_maps)
        final_pixel_scores = pixel_residual + batch_frozen_patch
        final_image_scores = image_residual + batch_baseline_scores
        batch_image_scores = final_image_scores.cpu().tolist()
        batch_pixel_scores = F.interpolate(
            final_pixel_scores,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).cpu()
        image_scores.extend(batch_image_scores)
        pixel_scores.extend(batch_pixel_scores[:, 0].numpy().astype(np.float32))
    return image_scores, pixel_scores


@torch.no_grad()
def measure_identity_drift(
    model: MapFusionHead,
    eval_maps: torch.Tensor,
    eval_frozen_patch_maps: torch.Tensor,
    eval_baseline_scores: torch.Tensor,
    baseline_pixel_scores: list[np.ndarray],
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    image_scores, pixel_scores = predict_dual_head(
        model=model,
        eval_maps=eval_maps,
        eval_frozen_patch_maps=eval_frozen_patch_maps,
        eval_baseline_scores=eval_baseline_scores,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    baseline_image = eval_baseline_scores.detach().cpu().numpy().astype(np.float32)
    image_abs = np.abs(np.asarray(image_scores, dtype=np.float32) - baseline_image)
    if pixel_scores:
        pixel_abs = np.concatenate(
            [
                np.abs(pred - baseline).reshape(-1)
                for pred, baseline in zip(pixel_scores, baseline_pixel_scores, strict=True)
            ],
            axis=0,
        )
    else:
        pixel_abs = np.zeros(1, dtype=np.float32)
    return {
        "max_abs_image_diff": float(image_abs.max()) if image_abs.size else 0.0,
        "mean_abs_image_diff": float(image_abs.mean() if image_abs.size else 0.0),
        "max_abs_pixel_diff": float(pixel_abs.max()) if pixel_abs.size else 0.0,
        "mean_abs_pixel_diff": float(pixel_abs.mean() if pixel_abs.size else 0.0),
    }


@torch.no_grad()
def measure_rank_activity(
    model: MapFusionHead,
    train_maps: torch.Tensor,
    train_labels: torch.Tensor,
    baseline_scores: torch.Tensor,
    calibration: dict[str, float],
    batch_size: int,
    margin: float,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        TensorDataset(train_maps, train_labels, baseline_scores),
        batch_size=batch_size,
        shuffle=False,
    )
    rank_relu_terms: list[float] = []
    rank_smooth_terms: list[float] = []
    for batch_maps, batch_labels, batch_baseline_scores in loader:
        batch_maps = batch_maps.to(device)
        batch_labels = batch_labels.to(device)
        batch_baseline_scores = batch_baseline_scores.to(device)
        _, image_residual = model(batch_maps)
        final_image_scores = image_residual + batch_baseline_scores
        image_logits = apply_score_calibration(final_image_scores, calibration)
        rank_relu_terms.append(float(pairwise_margin_rank(image_logits, batch_labels, margin=margin, smooth=False).item()))
        rank_smooth_terms.append(float(pairwise_margin_rank(image_logits, batch_labels, margin=margin, smooth=True).item()))
    return {
        "rank_relu": float(np.mean(rank_relu_terms)) if rank_relu_terms else 0.0,
        "rank_smooth": float(np.mean(rank_smooth_terms)) if rank_smooth_terms else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, experiments_rows: list[dict[str, object]], mode: str) -> None:
    lines = [f"# Stage3 {mode.upper()} summary", ""]
    for row in experiments_rows:
        lines.append(
            f"{row['experiment']}: image={row['image_auroc_mean']:.4f} "
            f"pixel={row['pixel_auroc_mean']:.4f} pro={row['pro_mean']:.4f} "
            f"balanced={row['balanced_mean']:.4f}"
        )
    if len(experiments_rows) >= 2:
        base = experiments_rows[0]
        cand = experiments_rows[-1]
        lines.append(
            f"delta: image={cand['image_auroc_mean'] - base['image_auroc_mean']:+.4f} "
            f"pixel={cand['pixel_auroc_mean'] - base['pixel_auroc_mean']:+.4f} "
            f"pro={cand['pro_mean'] - base['pro_mean']:+.4f} "
            f"balanced={cand['balanced_mean'] - base['balanced_mean']:+.4f}"
        )
    if mode == MODE_P1:
        lines.append("note: pixel/pro stay frozen from cached subspace maps in E1; only image scores are updated.")
    if mode == MODE_P2:
        lines.append("note: P2 metrics are evaluated in raw score space; BCE/rank use support-fitted affine score calibration.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, image_size = load_records(cache_dir)
    train_records, eval_records = split_records(records)

    baseline_image_scores = [float(record["winner_image_score"]) for record in eval_records]
    baseline_pixel_scores = [record["pixel_map"] for record in eval_records]
    baseline_rows = evaluate_per_category(
        eval_records=eval_records,
        image_scores=baseline_image_scores,
        pixel_scores=baseline_pixel_scores,
        image_size=image_size,
    )
    baseline_summary = aggregate_rows(
        baseline_rows,
        experiment="E0",
        pixel_source="frozen_subspace_cache",
        balanced_mode="frozen_image_plus_frozen_pixel",
    )

    per_category_rows = [{"experiment": "E0", **row} for row in baseline_rows]
    experiments_rows = [baseline_summary]
    history: list[dict[str, float]] = []
    checkpoint_name = "image_head.pt" if args.mode == MODE_P1 else "dual_head.pt"
    extra_stats_name = "feature_stats.json" if args.mode == MODE_P1 else "map_stats.json"
    extra_stats: dict[str, object]
    identity_check: dict[str, float] | None = None
    rank_check: dict[str, float] | None = None
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    if args.mode == MODE_P1:
        train_x, train_y, eval_x, extra_stats = normalize_features(train_records, eval_records)
        model = ImagePoolHead(
            in_features=int(train_x.shape[1]),
            hidden_features=args.hidden_features,
            dropout=args.dropout,
        ).to(device)
        history = train_image_head(
            model=model,
            train_x=train_x,
            train_y=train_y,
            args=args,
            device=device,
            metrics_path=metrics_path,
        )
        image_scores = predict_image_scores(model=model, features=eval_x, device=device)
        p1_rows = evaluate_per_category(
            eval_records=eval_records,
            image_scores=image_scores,
            pixel_scores=baseline_pixel_scores,
            image_size=image_size,
        )
        p1_summary = aggregate_rows(
            p1_rows,
            experiment="E1",
            pixel_source="frozen_subspace_cache",
            balanced_mode="learned_image_plus_frozen_pixel",
        )
        per_category_rows.extend({"experiment": "E1", **row} for row in p1_rows)
        experiments_rows.append(p1_summary)
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "feature_stats": extra_stats,
            "config": vars(args),
        }
    else:
        train_maps, eval_maps, extra_stats = normalize_map_tensors(train_records, eval_records)
        map_hw = (int(train_maps.shape[-2]), int(train_maps.shape[-1]))
        train_labels, train_masks, consistency_targets, consistency_selector, train_baseline_scores = build_dual_head_train_targets(
            train_records,
            output_hw=map_hw,
        )
        image_calibration = fit_image_score_calibration(train_baseline_scores, train_labels)
        pixel_calibration = fit_pixel_score_calibration(consistency_targets, train_masks)
        extra_stats["image_score_calibration"] = image_calibration
        extra_stats["pixel_score_calibration"] = pixel_calibration
        eval_frozen_patch_maps = torch.stack([record["frozen_patch_map"] for record in eval_records], dim=0)
        eval_baseline_scores = torch.tensor(
            [float(record["winner_image_score"]) for record in eval_records],
            dtype=torch.float32,
        )
        model = MapFusionHead(
            in_channels=int(train_maps.shape[1]),
            hidden_channels=args.map_hidden_channels,
            image_hidden_channels=args.image_hidden_channels,
            dropout=args.dropout,
            topk_ratio=args.map_topk_ratio,
        ).to(device)
        initialize_dual_head_as_identity(model)
        identity_check = measure_identity_drift(
            model=model,
            eval_maps=eval_maps,
            eval_frozen_patch_maps=eval_frozen_patch_maps,
            eval_baseline_scores=eval_baseline_scores,
            baseline_pixel_scores=baseline_pixel_scores,
            image_size=image_size,
            batch_size=args.batch_size,
            device=device,
        )
        extra_stats["identity_check"] = identity_check
        if (
            identity_check["max_abs_image_diff"] > IDENTITY_TOLERANCE
            or identity_check["max_abs_pixel_diff"] > IDENTITY_TOLERANCE
        ):
            raise RuntimeError(f"P2 identity contract drifted before training: {identity_check}")
        rank_check = measure_rank_activity(
            model=model,
            train_maps=train_maps,
            train_labels=train_labels,
            baseline_scores=train_baseline_scores,
            calibration=image_calibration,
            batch_size=args.batch_size,
            margin=args.margin,
            device=device,
        )
        extra_stats["initial_rank_activity"] = rank_check
        if rank_check["rank_smooth"] <= 0.0:
            raise RuntimeError(f"P2 rank objective is inactive before training: {rank_check}")
        (output_dir / "identity_check.json").write_text(json.dumps(identity_check, indent=2), encoding="utf-8")
        (output_dir / extra_stats_name).write_text(json.dumps(extra_stats, indent=2), encoding="utf-8")
        history, best_state = train_dual_head(
            model=model,
            train_maps=train_maps,
            train_labels=train_labels,
            train_masks=train_masks,
            consistency_targets=consistency_targets,
            consistency_selector=consistency_selector,
            baseline_scores=train_baseline_scores,
            image_calibration=image_calibration,
            pixel_calibration=pixel_calibration,
            args=args,
            device=device,
            metrics_path=metrics_path,
        )
        model.load_state_dict(best_state)
        image_scores, pixel_scores = predict_dual_head(
            model=model,
            eval_maps=eval_maps,
            eval_frozen_patch_maps=eval_frozen_patch_maps,
            eval_baseline_scores=eval_baseline_scores,
            image_size=image_size,
            batch_size=args.batch_size,
            device=device,
        )
        p2_rows = evaluate_per_category(
            eval_records=eval_records,
            image_scores=image_scores,
            pixel_scores=pixel_scores,
            image_size=image_size,
        )
        p2_summary = aggregate_rows(
            p2_rows,
            experiment="E2",
            pixel_source="learned_dual_head",
            balanced_mode="learned_image_plus_learned_pixel",
        )
        per_category_rows.extend({"experiment": "E2", **row} for row in p2_rows)
        experiments_rows.append(p2_summary)
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "map_stats": extra_stats,
            "identity_check": identity_check,
            "rank_check": rank_check,
            "config": vars(args),
        }

    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_summary(output_dir / "summary.md", experiments_rows=experiments_rows, mode=args.mode)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / extra_stats_name).write_text(json.dumps(extra_stats, indent=2), encoding="utf-8")
    if rank_check is not None:
        (output_dir / "rank_check.json").write_text(json.dumps(rank_check, indent=2), encoding="utf-8")
    torch.save(checkpoint_payload, output_dir / checkpoint_name)

    candidate_summary = experiments_rows[-1]
    result = {
        "mode": args.mode,
        "output_dir": str(output_dir),
        "baseline": experiments_rows[0],
        "candidate": candidate_summary,
        "delta_image_auroc": candidate_summary["image_auroc_mean"] - experiments_rows[0]["image_auroc_mean"],
        "delta_pixel_auroc": candidate_summary["pixel_auroc_mean"] - experiments_rows[0]["pixel_auroc_mean"],
        "delta_pro": candidate_summary["pro_mean"] - experiments_rows[0]["pro_mean"],
        "delta_balanced": candidate_summary["balanced_mean"] - experiments_rows[0]["balanced_mean"],
    }
    if identity_check is not None:
        result["identity_check"] = identity_check
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
