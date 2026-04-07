import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.cache import FeatureCacheSpec, collect_image_paths, load_feature_cache_batch, populate_feature_cache
from fewshot.data import (
    build_shared_split_manifest,
    load_mask_array,
    load_shared_split_manifest,
    save_shared_split_manifest,
)
from fewshot.feature_bank import PROTOTYPE_FAMILY_MEMORY_BANK, build_reference_bank, flatten_feature_map
from fewshot.scoring import AGGREGATION_MODE_MAX, aggregate_image_score, reference_similarity_map
from fewshot.stage_a1 import binary_auroc, pixel_auroc
from fewshot.subspace import SUBSPACE_MODE_LOCAL, subspace_score_map


DEFAULT_CATEGORIES = ("leather", "grid", "carpet", "screw", "zipper", "bottle")
VALUE_SPACE_CONTRACT = {
    "base_anomaly_map": "score",
    "subspace_residual_map": "score",
    "knn_top1_map": "score",
    "knn_top3_map": "score",
    "knn_gap_map": "score_delta",
    "image_score_base": "score",
    "winner_image_score": "score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Stage2 winner maps for Stage3 experiments.")
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--manifests-dir", default="outputs/split_manifests/stage2")
    parser.add_argument("--cache-root", default="outputs/stage3/cache")
    parser.add_argument("--subset-name", default="weak5_bottle")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support-normal-k", type=int, default=16)
    parser.add_argument("--support-defect-k", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--feature-layer", default="layer4")
    parser.add_argument("--reference-topk", type=int, default=3)
    parser.add_argument("--subspace-dim", type=int, default=8)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--aggregation-mode", default=AGGREGATION_MODE_MAX)
    parser.add_argument(
        "--stage2-run-dir",
        default=(
            "outputs/stage2/p4_full15_final/seed42/"
            "a1_sn016_sd004_layer4_neg_normal_subspace_refk003_max_patch_topk010_dim008"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manifest_path(root: Path, category: str, sn: int, sd: int, seed: int) -> Path:
    return root / f"{category}_sn{sn}_sd{sd}_seed{seed}.json"


def ensure_manifest(args: argparse.Namespace, category: str) -> Path:
    path = manifest_path(
        root=Path(args.manifests_dir),
        category=category,
        sn=args.support_normal_k,
        sd=args.support_defect_k,
        seed=args.seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    manifest = build_shared_split_manifest(
        root=args.data_root,
        category=category,
        support_normal_k=args.support_normal_k,
        support_defect_k=args.support_defect_k,
        seed=args.seed,
    )
    save_shared_split_manifest(manifest, path)
    return path


def build_cache_spec(args: argparse.Namespace, category: str) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        cache_root=Path("outputs/feature_cache") / category,
        image_size=args.image_size,
        pretrained=args.pretrained,
        feature_layer=args.feature_layer,
        seed=args.seed,
    )


def sample_slug(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def sample_role(manifest, sample) -> str:
    sample_path = str(sample.path)
    if any(str(item.path) == sample_path for item in manifest.support_normal):
        return "support_normal"
    if any(str(item.path) == sample_path for item in manifest.support_defect):
        return "support_defect"
    return "query_eval"


def upsample_map(score_map: torch.Tensor, image_size: int) -> np.ndarray:
    return (
        F.interpolate(score_map.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def pro_score(mask_arrays: list[np.ndarray], score_maps: list[np.ndarray], max_fpr: float = 0.3) -> float:
    if not mask_arrays or not score_maps:
        return float("nan")
    all_scores = np.concatenate([score.reshape(-1) for score in score_maps], axis=0)
    negative_scores = np.concatenate(
        [score[mask <= 0.5].reshape(-1) for mask, score in zip(mask_arrays, score_maps)],
        axis=0,
    )
    regions = []
    for mask in mask_arrays:
        labeled, num = ndimage.label(mask > 0.5)
        regions.append([(labeled == label_id) for label_id in range(1, num + 1)])
    if negative_scores.size == 0 or sum(len(items) for items in regions) == 0:
        return float("nan")
    thresholds = np.linspace(float(all_scores.max()), float(all_scores.min()), 200)
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


def evaluate_replay(args: argparse.Namespace, records: list[dict[str, object]], stage2_metrics_path: Path) -> dict[str, float]:
    query_records = [record for record in records if record["role"] == "query_eval"]
    image_labels = [int(record["label"]) for record in query_records]
    image_scores = [float(record["winner_image_score"]) for record in query_records]
    pixel_masks = [
        load_mask_array(record["mask_path"] or None, image_size=args.image_size)
        for record in query_records
    ]
    pixel_scores = []
    for record in query_records:
        payload = torch.load(record["cache_path"], map_location="cpu")
        pixel_scores.append(upsample_map(payload["subspace_residual_map"], args.image_size))
    replay = {
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
        "pro": pro_score(pixel_masks, pixel_scores),
    }
    stage2_metrics = json.loads(stage2_metrics_path.read_text(encoding="utf-8"))
    replay["stage2_image_auroc"] = float(stage2_metrics["image_auroc"])
    replay["stage2_pixel_auroc"] = float(stage2_metrics["pixel_auroc"])
    replay["stage2_pro"] = float(stage2_metrics["pro"])
    replay["image_auroc_diff"] = replay["image_auroc"] - replay["stage2_image_auroc"]
    replay["pixel_auroc_diff"] = replay["pixel_auroc"] - replay["stage2_pixel_auroc"]
    replay["pro_diff"] = replay["pro"] - replay["stage2_pro"]
    replay["max_abs_diff"] = max(abs(replay["image_auroc_diff"]), abs(replay["pixel_auroc_diff"]), abs(replay["pro_diff"]))
    return replay


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stage2_run_dir = Path(args.stage2_run_dir)
    output_root = Path(args.cache_root) / args.subset_name / f"seed{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    encoder = DenseClipVisualEncoder(
        pretrained=args.pretrained,
        input_resolution=args.image_size,
        freeze=True,
    ).to(device)
    encoder.eval()

    replay_rows: list[dict[str, object]] = []
    all_index_rows: list[dict[str, object]] = []
    for category in args.categories:
        manifest = load_shared_split_manifest(ensure_manifest(args, category))
        category_root = output_root / category
        category_root.mkdir(parents=True, exist_ok=True)
        save_shared_split_manifest(manifest, category_root / "split_manifest.json")

        spec = build_cache_spec(args, category)
        all_paths = collect_image_paths(manifest.support_normal + manifest.support_defect + manifest.query_eval)
        populate_feature_cache(encoder=encoder, image_paths=all_paths, spec=spec, device=device, batch_size=args.batch_size)
        support_features = load_feature_cache_batch(collect_image_paths(manifest.support_normal), spec=spec).to(device)
        normal_bank = build_reference_bank(
            features=flatten_feature_map(support_features),
            prototype_family=PROTOTYPE_FAMILY_MEMORY_BANK,
            num_prototypes=1,
            seed=args.seed,
            num_iters=1,
        ).to(device)

        category_rows: list[dict[str, object]] = []
        samples = list(manifest.support_normal) + list(manifest.support_defect) + list(manifest.query_eval)
        for sample in samples:
            role = sample_role(manifest, sample)
            query_feature = load_feature_cache_batch([sample.path], spec=spec).to(device)
            knn_top1 = -reference_similarity_map(query_feature, normal_bank, reference_topk=1)
            knn_top3 = -reference_similarity_map(query_feature, normal_bank, reference_topk=args.reference_topk)
            subspace_map = subspace_score_map(
                support_feature_batch=support_features,
                query_feature_batch=query_feature,
                subspace_dim=args.subspace_dim,
                mode=SUBSPACE_MODE_LOCAL,
            )
            payload = {
                "base_anomaly_map": knn_top3[0].detach().cpu().contiguous(),
                "subspace_residual_map": subspace_map[0].detach().cpu().contiguous(),
                "knn_top1_map": knn_top1[0].detach().cpu().contiguous(),
                "knn_top3_map": knn_top3[0].detach().cpu().contiguous(),
                "knn_gap_map": (knn_top1 - knn_top3)[0].detach().cpu().contiguous(),
                "image_score_base": float(aggregate_image_score(knn_top3, args.aggregation_mode, args.topk_ratio)[0].item()),
                "winner_image_score": float(aggregate_image_score(subspace_map, args.aggregation_mode, args.topk_ratio)[0].item()),
                "image_label": int(sample.label or 0),
                "category": category,
                "role": role,
                "source_path": str(sample.path),
                "defect_type": sample.defect_type,
                "split_manifest": str((category_root / "split_manifest.json").resolve()),
            }
            if role == "support_defect":
                payload["mask"] = torch.from_numpy(load_mask_array(sample.mask_path, args.image_size)).unsqueeze(0)
            elif role == "support_normal":
                payload["mask"] = torch.zeros((1, args.image_size, args.image_size), dtype=torch.float32)
            cache_path = category_root / f"{role}__{sample_slug(sample.path)}.pt"
            torch.save(payload, cache_path)
            row = {
                "category": category,
                "role": role,
                "path": str(sample.path),
                "cache_path": str(cache_path.resolve()),
                "label": int(sample.label or 0),
                "defect_type": sample.defect_type or "good",
                "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
                "image_score_base": payload["image_score_base"],
                "winner_image_score": payload["winner_image_score"],
            }
            category_rows.append(row)
            all_index_rows.append(row)

        replay = evaluate_replay(
            args=args,
            records=category_rows,
            stage2_metrics_path=stage2_run_dir / category / "metrics.json",
        )
        replay_row = {"category": category, **replay}
        replay_rows.append(replay_row)
        (category_root / "cache_index.json").write_text(json.dumps(category_rows, indent=2), encoding="utf-8")
        (category_root / "replay_metrics.json").write_text(json.dumps(replay_row, indent=2), encoding="utf-8")

    summary = {
        "subset_name": args.subset_name,
        "seed": args.seed,
        "categories": list(args.categories),
        "rows": replay_rows,
        "max_abs_diff": max(row["max_abs_diff"] for row in replay_rows) if replay_rows else float("nan"),
        "passed_replay_check": all(row["max_abs_diff"] <= 0.002 for row in replay_rows),
    }
    config = {
        **vars(args),
        "value_space_contract": VALUE_SPACE_CONTRACT,
    }
    (output_root / "index.json").write_text(json.dumps(all_index_rows, indent=2), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_root / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_root / "replay_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0].keys()) if replay_rows else ["category"])
        writer.writeheader()
        for row in replay_rows:
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
