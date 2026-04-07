import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fewshot.data import (
    StageA1Dataset,
    build_shared_split_manifest,
    load_image_rgb,
    load_shared_split_manifest,
    save_shared_split_manifest,
    stage_b_split_from_manifest,
)
from fewshot.feature_bank import build_prototype_bank
from fewshot.learned_head import LearnedHeadAnomalyModel
from fewshot.stage_a1 import binary_auroc, pixel_auroc, save_prediction_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage B learned-head comparison on official MVTec structure.")
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--output-dir", default="outputs/stage_b")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--support-normal-k", type=int, default=8)
    parser.add_argument("--support-defect-k", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def save_outputs(
    output_root: Path,
    split,
    args: argparse.Namespace,
    metrics: dict[str, float],
    predictions: list[dict[str, object]],
    checkpoint_path: Path,
    train_history: list[dict[str, float]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    support_paths = {
        "support_normal": [str(sample.path) for sample in split.support_normal],
        "support_defect": [str(sample.path) for sample in split.support_defect],
    }
    (output_root / "support_paths.json").write_text(json.dumps(support_paths, indent=2), encoding="utf-8")
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output_root / "train_history.json").write_text(json.dumps(train_history, indent=2), encoding="utf-8")
    (output_root / "checkpoint_summary.json").write_text(
        json.dumps({"checkpoint_path": str(checkpoint_path)}, indent=2),
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


def train_model(
    model: LearnedHeadAnomalyModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    optimizer = AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device=device, dtype=torch.float32)
            outputs = model(images)
            loss = F.binary_cross_entropy_with_logits(outputs["image_logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"epoch={epoch} train_loss={mean_loss:.4f}")
    return history


@torch.no_grad()
def evaluate_model(
    model: LearnedHeadAnomalyModel,
    loader: DataLoader,
    image_size: int,
    output_root: Path,
    save_visuals: bool,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    device = next(model.parameters()).device
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_masks: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    predictions: list[dict[str, object]] = []

    for batch in loader:
        outputs = model(batch["image"].to(device))
        batch_image_scores = outputs["image_scores"].cpu().tolist()
        batch_maps = outputs["upsampled_scores"].cpu().numpy()
        batch_masks = batch["mask"].numpy()

        for index, score in enumerate(batch_image_scores):
            label = int(batch["label"][index])
            score_map = batch_maps[index, 0].astype(np.float32)
            mask_array = batch_masks[index, 0].astype(np.float32)
            image_labels.append(label)
            image_scores.append(float(score))
            pixel_masks.append(mask_array)
            pixel_scores.append(score_map)

            prediction = {
                "path": batch["path"][index],
                "label": label,
                "defect_type": batch["defect_type"][index],
                "image_score": float(score),
                "mask_path": batch["mask_path"][index],
                "heatmap_path": "",
                "overlay_path": "",
            }
            if save_visuals:
                artifacts = save_prediction_artifacts(
                    image_path=batch["path"][index],
                    image_rgb=load_image_rgb(batch["path"][index], image_size=image_size),
                    score_map=score_map,
                    output_dir=output_root,
                )
                prediction.update(artifacts)
            predictions.append(prediction)

    metrics = {
        "image_auroc": binary_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
    }
    return metrics, predictions


@torch.no_grad()
def save_checkpoint(
    output_root: Path,
    model: LearnedHeadAnomalyModel,
    split,
    args: argparse.Namespace,
    manifest_path: Path,
) -> Path:
    checkpoint_path = output_root / "stage_b_learned_head.pt"
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "normal_prototype": model.normal_prototype.detach().cpu(),
            "defect_prototype": model.defect_prototype.detach().cpu(),
            "support_paths": {
                "support_normal": [str(sample.path) for sample in split.support_normal],
                "support_defect": [str(sample.path) for sample in split.support_defect],
            },
            "config": vars(args),
            "split_manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    if args.support_defect_k < 1:
        raise ValueError("Stage B learned-head requires --support-defect-k >= 1.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir) / args.category
    output_root.mkdir(parents=True, exist_ok=True)

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
    split = stage_b_split_from_manifest(manifest)
    if not split.support_defect:
        raise ValueError("Stage B learned-head requires a split manifest with at least one defect support sample.")
    save_shared_split_manifest(manifest, output_root / "split_manifest.json")

    train_samples = split.support_normal + split.support_defect
    train_dataset = StageA1Dataset(train_samples, image_size=args.image_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    query_dataset = StageA1Dataset(split.test_query, image_size=args.image_size, augment=False)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = LearnedHeadAnomalyModel(
        pretrained=args.pretrained,
        image_size=args.image_size,
        hidden_channels=args.hidden_channels,
        topk_ratio=args.topk_ratio,
        dropout=args.dropout,
    ).to(device)
    bank = build_prototype_bank(
        encoder=model.encoder,
        support_normal=split.support_normal,
        support_defect=split.support_defect,
        image_size=args.image_size,
        device=device,
    )
    model.set_prototype_bank(bank)

    print(f"Using device: {device}")
    print(f"Support normal: {len(split.support_normal)}, support defect: {len(split.support_defect)}")
    print(f"Evaluation query: {len(split.test_query)}")

    train_history = train_model(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    checkpoint_path = save_checkpoint(
        output_root=output_root,
        model=model,
        split=split,
        args=args,
        manifest_path=manifest_path,
    )
    metrics, predictions = evaluate_model(
        model=model,
        loader=query_loader,
        image_size=args.image_size,
        output_root=output_root,
        save_visuals=args.save_visuals,
    )
    metrics.update(
        {
            "category": args.category,
            "num_support_normal": len(split.support_normal),
            "num_support_defect": len(split.support_defect),
            "num_test_query": len(split.test_query),
        }
    )
    save_outputs(
        output_root=output_root,
        split=split,
        args=args,
        metrics=metrics,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
        train_history=train_history,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
