import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fewshot.backbone import DenseClipVisualEncoder
from fewshot.data import (
    StageA1Dataset,
    build_shared_split_manifest,
    load_image_rgb,
    load_shared_split_manifest,
    save_shared_split_manifest,
    stage_a1_split_from_manifest,
)
from fewshot.feature_bank import (
    PROTOTYPE_FAMILIES,
    PROTOTYPE_FAMILY_MEAN,
    build_prototype_bank,
)
from fewshot.scoring import (
    AGGREGATION_MODE_TOPK_MEAN,
    AGGREGATION_MODES,
    AGGREGATION_STAGE_UPSAMPLED,
    AGGREGATION_STAGES,
    FEATURE_LAYER_LOCAL,
    FEATURE_LAYERS,
    SCORE_MODE_NEG_NORMAL,
    SCORE_MODE_ONE_MINUS_NORMAL,
)
from fewshot.stage_a1 import binary_auroc, pixel_auroc, save_prediction_artifacts, score_with_normal_prototype


A1_SCORE_MODES = (
    SCORE_MODE_ONE_MINUS_NORMAL,
    SCORE_MODE_NEG_NORMAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A1 visual-prototype baseline on official MVTec structure.")
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--output-dir", default="outputs/stage_a1")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--support-normal-k", type=int, default=8)
    parser.add_argument("--support-defect-k", type=int, default=0)
    parser.add_argument("--score-mode", default=SCORE_MODE_ONE_MINUS_NORMAL, choices=A1_SCORE_MODES)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--aggregation-mode", default=AGGREGATION_MODE_TOPK_MEAN, choices=AGGREGATION_MODES)
    parser.add_argument("--aggregation-stage", default=AGGREGATION_STAGE_UPSAMPLED, choices=AGGREGATION_STAGES)
    parser.add_argument("--feature-layer", default=FEATURE_LAYER_LOCAL, choices=FEATURE_LAYERS)
    parser.add_argument("--prototype-family", default=PROTOTYPE_FAMILY_MEAN, choices=PROTOTYPE_FAMILIES)
    parser.add_argument("--num-prototypes", type=int, default=1)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--reference-topk", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_outputs(
    output_root: Path,
    split,
    args: argparse.Namespace,
    metrics: dict[str, float],
    predictions: list[dict[str, object]],
    prototype_path: Path,
    num_normal_references: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    support_paths = {
        "support_normal": [str(sample.path) for sample in split.support_normal],
    }
    (output_root / "support_paths.json").write_text(json.dumps(support_paths, indent=2), encoding="utf-8")
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output_root / "prototype_summary.json").write_text(
        json.dumps(
            {
                "prototype_path": str(prototype_path),
                "num_support_normal": len(split.support_normal),
                "prototype_family": args.prototype_family,
                "num_prototypes": args.num_prototypes,
                "reference_topk": args.reference_topk,
                "num_normal_references": num_normal_references,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_path = output_root / "predictions.csv"
    fieldnames = [
        "path",
        "label",
        "defect_type",
        "image_score",
        "mask_path",
        "heatmap_path",
        "overlay_path",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir) / args.category

    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
        manifest = load_shared_split_manifest(manifest_path)
        if manifest.category != args.category:
            raise ValueError(
                f"Split manifest category mismatch: expected {args.category}, found {manifest.category}"
            )
    else:
        manifest = build_shared_split_manifest(
            root=args.data_root,
            category=args.category,
            support_normal_k=args.support_normal_k,
            support_defect_k=args.support_defect_k,
            seed=args.seed,
        )
        manifest_path = output_root / "split_manifest.json"
    split = stage_a1_split_from_manifest(manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    save_shared_split_manifest(manifest, output_root / "split_manifest.json")
    dataset = StageA1Dataset(split.test_query, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    encoder = DenseClipVisualEncoder(
        pretrained=args.pretrained,
        input_resolution=args.image_size,
        freeze=True,
    ).to(device)
    encoder.eval()

    bank = build_prototype_bank(
        encoder=encoder,
        support_normal=split.support_normal,
        support_defect=[],
        image_size=args.image_size,
        device=device,
        feature_layer=args.feature_layer,
        prototype_family=args.prototype_family,
        num_prototypes=args.num_prototypes,
        seed=args.seed,
        kmeans_iters=args.kmeans_iters,
    )
    normal_prototype = bank.normal_reference
    prototype_path = output_root / "stage_a1_prototype.pt"
    torch.save(
        {
            "normal_prototype": normal_prototype.detach().cpu(),
            "normal_reference": normal_prototype.detach().cpu(),
            "support_normal_paths": bank.normal_support_paths,
            "prototype_family": args.prototype_family,
            "num_prototypes": args.num_prototypes,
            "reference_topk": args.reference_topk,
            "config": vars(args),
            "split_manifest_path": str(manifest_path),
        },
        prototype_path,
    )

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_masks: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    predictions: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device)
        outputs = score_with_normal_prototype(
            encoder=encoder,
            images=images,
            normal_prototype=normal_prototype.to(device),
            image_size=args.image_size,
            topk_ratio=args.topk_ratio,
            score_mode=args.score_mode,
            aggregation_mode=args.aggregation_mode,
            aggregation_stage=args.aggregation_stage,
            feature_layer=args.feature_layer,
            reference_topk=args.reference_topk,
        )

        batch_scores = outputs["image_scores"].cpu().tolist()
        batch_maps = outputs["upsampled_map"].cpu().numpy()
        batch_masks = batch["mask"].numpy()

        for index, score in enumerate(batch_scores):
            label = int(batch["label"][index])
            score_map = batch_maps[index, 0]
            mask_array = batch_masks[index, 0].astype(np.float32)
            image_labels.append(label)
            image_scores.append(float(score))
            pixel_masks.append(mask_array)
            pixel_scores.append(score_map.astype(np.float32))

            prediction = {
                "path": batch["path"][index],
                "label": label,
                "defect_type": batch["defect_type"][index],
                "image_score": float(score),
                "mask_path": batch["mask_path"][index],
                "heatmap_path": "",
                "overlay_path": "",
            }
            if args.save_visuals:
                artifacts = save_prediction_artifacts(
                    image_path=batch["path"][index],
                    image_rgb=load_image_rgb(batch["path"][index], image_size=args.image_size),
                    score_map=score_map,
                    output_dir=output_root,
                )
                prediction.update(artifacts)
            predictions.append(prediction)

    metrics = {
        "category": args.category,
        "score_mode": args.score_mode,
        "num_support_normal": len(split.support_normal),
        "num_test_query": len(split.test_query),
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
        "reference_topk": args.reference_topk,
    }
    save_outputs(
        output_root=output_root,
        split=split,
        args=args,
        metrics=metrics,
        predictions=predictions,
        prototype_path=prototype_path,
        num_normal_references=int(normal_prototype.shape[0]) if normal_prototype.ndim == 2 else 1,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
