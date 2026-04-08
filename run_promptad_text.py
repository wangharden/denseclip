import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_ROOT = REPO_ROOT / "detection" / "denseclip"
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

from untils import tokenize

from run_prompt_context_text import (
    CLIPTextContextEncoder,
    CONTROL_CATEGORIES,
    normalize_scores_by_category,
    sanitize_defect_name,
    score_stats_by_category,
)
from run_prompt_defect_text import (
    append_jsonl,
    append_result_rows,
    build_feature_matrix,
    build_predictions_rows,
    build_resplits,
    choose_thresholds_by_category,
    encode_clip_global_features,
    evaluate_rows,
    mean_score_lists,
    write_summary,
)
from run_stage3_head import load_records, resolve_device, split_records


DEFAULT_NORMAL_TEMPLATES = (
    "a photo of a normal {category}.",
    "a photo of a good {category}.",
    "a photo of a flawless {category}.",
    "a photo of a clean {category}.",
)
DEFAULT_ANOMALY_STATE_TEMPLATES = (
    "a photo of a defective {category}.",
    "a photo of a damaged {category}.",
    "a photo of an anomalous {category}.",
    "a flawed {category}.",
)
DEFAULT_SPECIFIC_DEFECT_TEMPLATES = (
    "a photo of a {category} with {defect}.",
    "a defective {category} showing {defect}.",
    "a damaged {category} with {defect}.",
)
DEFAULT_STATE_PROMPTS = (
    "a {category} with cracks.",
    "a {category} with scratches.",
    "a {category} with holes.",
    "a stained {category}.",
    "a contaminated {category}.",
    "a deformed {category}.",
    "a misaligned {category}.",
    "a bent {category}.",
    "a broken {category}.",
    "a {category} with a missing part.",
    "a discolored {category}.",
    "a cut {category}.",
)
MID_TIER_CATEGORIES = ("grid", "screw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PromptAD-like category-specific prompt learner on frozen CLIP image features."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs/new-branch/prompt-text/weak5_bottle/seed42/promptad_ctx_v1_screen",
    )
    parser.add_argument("--scope", default="")
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--num-resplits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prompt-token-length", type=int, default=24)
    parser.add_argument("--n-ctx", type=int, default=8)
    parser.add_argument("--pool-temperature", type=float, default=0.10)
    parser.add_argument("--context-l2", type=float, default=5e-4)
    parser.add_argument("--control-regularization-scale", type=float, default=4.0)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--margin-weight", type=float, default=0.6)
    parser.add_argument("--defect-bce-weight", type=float, default=0.35)
    parser.add_argument("--defect-margin-weight", type=float, default=0.35)
    parser.add_argument("--aux-defect-weight", type=float, default=0.08)
    parser.add_argument("--use-state-bank", action="store_true")
    parser.add_argument("--state-context-l2", type=float, default=5e-4)
    parser.add_argument("--use-category-state-gate", action="store_true")
    parser.add_argument("--state-gate-init-logit", type=float, default=-1.5)
    parser.add_argument("--state-gate-l2", type=float, default=1e-3)
    parser.add_argument("--control-state-gate-weight", type=float, default=5e-2)
    parser.add_argument("--control-loss-weight", type=float, default=1.0)
    parser.add_argument("--mid-defect-loss-weight", type=float, default=1.0)
    parser.add_argument("--control-taxonomy-only", action="store_true")
    parser.add_argument("--control-taxonomy-mix-alpha", type=float, default=0.0)
    parser.add_argument("--control-selection-weight", type=float, default=0.0)
    parser.add_argument("--mid-selection-weight", type=float, default=0.0)
    parser.add_argument("--control-anomaly-anchor-weight", type=float, default=0.0)
    parser.add_argument("--control-specific-overreach-weight", type=float, default=0.0)
    parser.add_argument("--control-specific-overreach-margin", type=float, default=0.0)
    parser.add_argument("--control-defect-overreach-weight", type=float, default=0.0)
    parser.add_argument("--generic-only-categories", default="")
    parser.add_argument("--variant-tag", default="promptad_ctx_v1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logsumexp_pool(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.logsumexp(values / temperature, dim=0) * temperature


def selection_metric(aggregate: dict[str, float]) -> float:
    return float(
        (
            float(aggregate["image_auroc_mean"])
            + float(aggregate["image_ap_mean"])
            + float(aggregate["balanced_accuracy"])
        )
        / 3.0
    )


def selection_metric_with_category_bonus(
    aggregate: dict[str, float],
    per_category: list[dict[str, object]],
    control_weight: float,
    mid_weight: float,
) -> float:
    value = selection_metric(aggregate)
    if not per_category:
        return value
    control_terms: list[float] = []
    mid_terms: list[float] = []
    for row in per_category:
        category = str(row["category"])
        category_value = (
            float(row["image_auroc"])
            + float(row["image_ap"])
            + float(row["balanced_accuracy"])
        ) / 3.0
        if category in CONTROL_CATEGORIES:
            control_terms.append(category_value)
        if category in MID_TIER_CATEGORIES:
            mid_terms.append(category_value)
    if control_terms:
        value += float(control_weight) * float(sum(control_terms) / len(control_terms))
    if mid_terms:
        value += float(mid_weight) * float(sum(mid_terms) / len(mid_terms))
    return value


def parse_category_list(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prompt_spec(
    train_records: list[dict[str, object]],
    prompt_token_length: int,
) -> dict[str, dict[str, object]]:
    defect_types_by_category: dict[str, set[str]] = defaultdict(set)
    for record in train_records:
        category = str(record["category"])
        defect_type = str(record.get("defect_type") or "").strip()
        if defect_type and defect_type != "good":
            defect_types_by_category[category].add(defect_type)

    prompt_spec: dict[str, dict[str, object]] = {}
    for category in sorted({str(record["category"]) for record in train_records}):
        normal_prompts = [template.format(category=category) for template in DEFAULT_NORMAL_TEMPLATES]
        anomaly_prompts = [template.format(category=category) for template in DEFAULT_ANOMALY_STATE_TEMPLATES]
        specific_prompts: dict[str, list[str]] = {}
        for defect_type in sorted(defect_types_by_category.get(category, set())):
            defect_name = sanitize_defect_name(defect_type)
            specific_prompts[defect_type] = [
                template.format(category=category, defect=defect_name)
                for template in DEFAULT_SPECIFIC_DEFECT_TEMPLATES
            ]
        prompt_spec[category] = {
            "normal_prompts": normal_prompts,
            "anomaly_prompts": anomaly_prompts,
            "state_prompts": [template.format(category=category) for template in DEFAULT_STATE_PROMPTS],
            "specific_defect_prompts": specific_prompts,
            "normal_tokens": tokenize(normal_prompts, context_length=prompt_token_length, truncate=True),
            "anomaly_tokens": tokenize(anomaly_prompts, context_length=prompt_token_length, truncate=True),
            "state_tokens": tokenize(
                [template.format(category=category) for template in DEFAULT_STATE_PROMPTS],
                context_length=prompt_token_length,
                truncate=True,
            ),
            "specific_defect_tokens": {
                defect_type: tokenize(prompts, context_length=prompt_token_length, truncate=True)
                for defect_type, prompts in specific_prompts.items()
            },
        }
    return prompt_spec


def encode_category_bank(
    category: str,
    prompt_spec: dict[str, dict[str, object]],
    text_encoder: CLIPTextContextEncoder,
    normal_context: torch.Tensor,
    anomaly_context: torch.Tensor,
    state_context: torch.Tensor | None,
    control_taxonomy_only: bool,
    control_taxonomy_mix_alpha: float,
    generic_only_categories: set[str],
    device: torch.device,
) -> dict[str, object]:
    spec = prompt_spec[category]
    normal_embeddings = text_encoder(spec["normal_tokens"].to(device), normal_context.unsqueeze(0))[0]
    anomaly_embeddings = text_encoder(spec["anomaly_tokens"].to(device), anomaly_context.unsqueeze(0))[0]
    normal_bank = F.normalize(normal_embeddings, dim=-1)
    generic_anomaly_bank = F.normalize(anomaly_embeddings, dim=-1)
    if state_context is not None:
        state_embeddings = text_encoder(spec["state_tokens"].to(device), state_context.unsqueeze(0))[0]
        state_bank = F.normalize(state_embeddings, dim=-1)
    else:
        state_bank = torch.empty((0, generic_anomaly_bank.shape[1]), device=device)

    specific_names: list[str] = []
    specific_slices: dict[str, tuple[int, int]] = {}
    specific_chunks: list[torch.Tensor] = []
    cursor = 0
    for defect_type, tokens in spec["specific_defect_tokens"].items():
        prompt_embeddings = text_encoder(tokens.to(device), anomaly_context.unsqueeze(0))[0]
        prompt_bank = F.normalize(prompt_embeddings, dim=-1)
        specific_names.append(defect_type)
        specific_slices[defect_type] = (cursor, cursor + prompt_bank.shape[0])
        specific_chunks.append(prompt_bank)
        cursor += prompt_bank.shape[0]

    if specific_chunks:
        specific_bank = torch.cat(specific_chunks, dim=0)
    else:
        specific_bank = torch.empty((0, generic_anomaly_bank.shape[1]), device=device)
    force_generic_only = category in generic_only_categories
    use_taxonomy_only = control_taxonomy_only and category in CONTROL_CATEGORIES and specific_bank.numel() > 0
    use_control_taxonomy_mix = (
        (not force_generic_only)
        and
        (not use_taxonomy_only)
        and category in CONTROL_CATEGORIES
        and specific_bank.numel() > 0
        and float(control_taxonomy_mix_alpha) > 0.0
    )
    if force_generic_only:
        base_anomaly_bank = generic_anomaly_bank
        specific_offset = generic_anomaly_bank.shape[0]
    elif use_taxonomy_only:
        base_anomaly_bank = specific_bank
        specific_offset = 0
    else:
        base_anomaly_bank = torch.cat([generic_anomaly_bank, specific_bank], dim=0)
        specific_offset = generic_anomaly_bank.shape[0]
    all_anomaly_bank = torch.cat([base_anomaly_bank, state_bank], dim=0)

    return {
        "normal_bank": normal_bank,
        "generic_anomaly_bank": generic_anomaly_bank,
        "state_bank": state_bank,
        "specific_anomaly_bank": specific_bank,
        "specific_defect_names": specific_names,
        "specific_slices": specific_slices,
        "base_anomaly_bank": base_anomaly_bank,
        "specific_offset": specific_offset,
        "all_anomaly_bank": all_anomaly_bank,
        "use_taxonomy_only": use_taxonomy_only,
        "use_control_taxonomy_mix": use_control_taxonomy_mix,
        "control_taxonomy_mix_alpha": float(control_taxonomy_mix_alpha),
        "force_generic_only": force_generic_only,
    }


def compute_batch_outputs(
    features: torch.Tensor,
    categories: list[str],
    defect_types: list[str],
    text_encoder: CLIPTextContextEncoder,
    prompt_spec: dict[str, dict[str, object]],
    normal_contexts: torch.Tensor,
    anomaly_contexts: torch.Tensor,
    state_context: torch.Tensor | None,
    state_gate_logits: torch.Tensor | None,
    control_taxonomy_only: bool,
    control_taxonomy_mix_alpha: float,
    generic_only_categories: set[str],
    category_to_index: dict[str, int],
    scale: torch.Tensor,
    pool_temperature: float,
) -> dict[str, object]:
    cache: dict[str, dict[str, object]] = {}
    anomaly_scores: list[torch.Tensor] = []
    normal_cos_values: list[torch.Tensor] = []
    anomaly_cos_values: list[torch.Tensor] = []
    generic_anomaly_cos_values: list[torch.Tensor] = []
    specific_anomaly_cos_values: list[torch.Tensor] = []
    target_anomaly_cos_values: list[torch.Tensor] = []
    aux_logits: list[torch.Tensor] = []
    aux_targets: list[int] = []
    details: list[dict[str, object]] = []

    for feature, category, defect_type in zip(features, categories, defect_types, strict=True):
        category_index = category_to_index[category]
        if category not in cache:
            cache[category] = encode_category_bank(
                category=category,
                prompt_spec=prompt_spec,
                text_encoder=text_encoder,
                normal_context=normal_contexts[category_index],
                anomaly_context=anomaly_contexts[category_index],
                state_context=state_context,
                control_taxonomy_only=control_taxonomy_only,
                control_taxonomy_mix_alpha=control_taxonomy_mix_alpha,
                generic_only_categories=generic_only_categories,
                device=feature.device,
            )
        bank = cache[category]
        feature = F.normalize(feature.unsqueeze(0), dim=-1)[0]
        normal_prompt_cos = torch.matmul(bank["normal_bank"], feature)
        generic_anomaly_cos = torch.matmul(bank["generic_anomaly_bank"], feature)
        specific_anomaly_cos = (
            torch.matmul(bank["specific_anomaly_bank"], feature)
            if bank["specific_anomaly_bank"].numel() > 0
            else torch.empty((0,), device=feature.device)
        )
        generic_pooled_anomaly = logsumexp_pool(generic_anomaly_cos, temperature=pool_temperature)
        if specific_anomaly_cos.numel() > 0:
            specific_pooled_anomaly = logsumexp_pool(specific_anomaly_cos, temperature=pool_temperature)
        else:
            specific_pooled_anomaly = generic_pooled_anomaly
        if bank["force_generic_only"]:
            base_anomaly_cos = generic_anomaly_cos
        elif bank["use_taxonomy_only"]:
            base_anomaly_cos = specific_anomaly_cos
        elif bank["use_control_taxonomy_mix"] and specific_anomaly_cos.numel() > 0:
            alpha = float(bank["control_taxonomy_mix_alpha"])
            generic_bias = math.log(max(1.0 - alpha, 1e-4))
            specific_bias = math.log(max(alpha, 1e-4))
            base_anomaly_cos = torch.cat(
                [generic_anomaly_cos + generic_bias, specific_anomaly_cos + specific_bias],
                dim=0,
            )
        else:
            base_anomaly_cos = torch.cat([generic_anomaly_cos, specific_anomaly_cos], dim=0)
        state_prompt_cos = (
            torch.matmul(bank["state_bank"], feature)
            if bank["state_bank"].numel() > 0
            else torch.empty((0,), device=feature.device)
        )
        if state_gate_logits is not None and state_prompt_cos.numel() > 0:
            state_mix = torch.sigmoid(state_gate_logits[category_index]).clamp(min=1e-4, max=1 - 1e-4)
            base_bias = torch.log1p(-state_mix)
            state_bias = torch.log(state_mix)
            anomaly_prompt_cos = torch.cat([base_anomaly_cos + base_bias, state_prompt_cos + state_bias], dim=0)
        else:
            anomaly_prompt_cos = torch.cat([base_anomaly_cos, state_prompt_cos], dim=0)
        pooled_normal = logsumexp_pool(normal_prompt_cos, temperature=pool_temperature)
        pooled_anomaly = logsumexp_pool(anomaly_prompt_cos, temperature=pool_temperature)
        anomaly_scores.append(scale * (pooled_anomaly - pooled_normal))
        normal_cos_values.append(pooled_normal)
        anomaly_cos_values.append(pooled_anomaly)
        generic_anomaly_cos_values.append(generic_pooled_anomaly)
        specific_anomaly_cos_values.append(specific_pooled_anomaly)

        target_anomaly = pooled_anomaly
        if (
            not bank["force_generic_only"]
            and defect_type
            and defect_type != "good"
            and defect_type in bank["specific_slices"]
        ):
            start, end = bank["specific_slices"][defect_type]
            offset = int(bank["specific_offset"])
            specific_logits = scale * anomaly_prompt_cos[offset + start : offset + end]
            if specific_logits.numel() > 0:
                aux_logits.append(specific_logits)
                aux_targets.append(0)
                target_anomaly = torch.max(anomaly_prompt_cos[offset + start : offset + end])
        target_anomaly_cos_values.append(target_anomaly)

        max_prompt_value, max_prompt_index = torch.max(anomaly_prompt_cos, dim=0)
        details.append(
            {
                "normal_score": float((scale * pooled_normal).detach().cpu().item()),
                "max_defect_score": float((scale * max_prompt_value).detach().cpu().item()),
                "top1_prompt_score": float((scale * max_prompt_value).detach().cpu().item()),
                "top1_prompt_index": int(max_prompt_index.detach().cpu().item()),
                "state_mix": (
                    None
                    if state_gate_logits is None or state_prompt_cos.numel() == 0
                    else float(torch.sigmoid(state_gate_logits[category_index]).detach().cpu().item())
                ),
            }
        )

    return {
        "anomaly_scores": torch.stack(anomaly_scores, dim=0),
        "normal_cos": torch.stack(normal_cos_values, dim=0),
        "anomaly_cos": torch.stack(anomaly_cos_values, dim=0),
        "generic_anomaly_cos": torch.stack(generic_anomaly_cos_values, dim=0),
        "specific_anomaly_cos": torch.stack(specific_anomaly_cos_values, dim=0),
        "target_anomaly_cos": torch.stack(target_anomaly_cos_values, dim=0),
        "aux_logits": aux_logits,
        "aux_targets": aux_targets,
        "details": details,
    }


def train_split_model(
    split_index: int,
    train_records: list[dict[str, object]],
    holdout_records: list[dict[str, object]],
    features_by_path: dict[str, torch.Tensor],
    prompt_spec: dict[str, dict[str, object]],
    pretrained: str,
    device: torch.device,
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict[str, object]:
    sample_tokens = next(iter(prompt_spec.values()))["normal_tokens"]
    prompt_token_length = int(sample_tokens.shape[1])
    total_context_length = prompt_token_length + args.n_ctx
    text_encoder = CLIPTextContextEncoder(
        context_length=total_context_length,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        embed_dim=1024,
        pretrained=pretrained,
    ).to(device)
    text_encoder.eval()

    category_names = sorted(prompt_spec.keys())
    category_to_index = {category: index for index, category in enumerate(category_names)}
    generic_only_categories = parse_category_list(args.generic_only_categories)
    pretrained_state = torch.jit.load(pretrained, map_location="cpu").float().state_dict()
    scale = torch.tensor(float(pretrained_state["logit_scale"].exp().item()), dtype=torch.float32, device=device)
    normal_contexts = nn.Parameter(torch.zeros(len(category_names), args.n_ctx, 512, device=device))
    anomaly_contexts = nn.Parameter(torch.zeros(len(category_names), args.n_ctx, 512, device=device))
    state_context: nn.Parameter | None = None
    if args.use_state_bank:
        state_context = nn.Parameter(torch.zeros(args.n_ctx, 512, device=device))
    state_gate_logits: nn.Parameter | None = None
    if args.use_state_bank and args.use_category_state_gate:
        state_gate_logits = nn.Parameter(
            torch.full((len(category_names),), float(args.state_gate_init_logit), device=device)
        )
    nn.init.normal_(normal_contexts, std=0.02)
    nn.init.normal_(anomaly_contexts, std=0.02)
    if state_context is not None:
        nn.init.normal_(state_context, std=0.02)
    optimizer_params: list[torch.Tensor] = [normal_contexts, anomaly_contexts]
    if state_context is not None:
        optimizer_params.append(state_context)
    if state_gate_logits is not None:
        optimizer_params.append(state_gate_logits)
    optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    train_features = build_feature_matrix(train_records, features_by_path).to(device)
    train_labels = torch.tensor([float(record["label"]) for record in train_records], dtype=torch.float32, device=device)
    holdout_features = build_feature_matrix(holdout_records, features_by_path).to(device)
    train_categories = [str(record["category"]) for record in train_records]
    train_defects = [str(record.get("defect_type") or "") for record in train_records]
    holdout_categories = [str(record["category"]) for record in holdout_records]
    holdout_defects = [str(record.get("defect_type") or "") for record in holdout_records]

    category_penalty_weight = torch.ones(len(category_names), device=device)
    for name in CONTROL_CATEGORIES:
        if name in category_to_index:
            category_penalty_weight[category_to_index[name]] = float(args.control_regularization_scale)
    train_normal_loss_weight = torch.ones(len(train_categories), dtype=torch.float32, device=device)
    train_defect_loss_weight = torch.ones(len(train_categories), dtype=torch.float32, device=device)
    for idx, category in enumerate(train_categories):
        if category in CONTROL_CATEGORIES:
            train_normal_loss_weight[idx] = float(args.control_loss_weight)
            train_defect_loss_weight[idx] = float(args.control_loss_weight)
        elif category in MID_TIER_CATEGORIES:
            train_defect_loss_weight[idx] = float(args.mid_defect_loss_weight)

    best_state = {
        "normal_contexts": normal_contexts.detach().cpu().clone(),
        "anomaly_contexts": anomaly_contexts.detach().cpu().clone(),
        "state_context": None if state_context is None else state_context.detach().cpu().clone(),
        "state_gate_logits": None if state_gate_logits is None else state_gate_logits.detach().cpu().clone(),
    }
    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    alerts: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        outputs = compute_batch_outputs(
            features=train_features,
            categories=train_categories,
            defect_types=train_defects,
            text_encoder=text_encoder,
            prompt_spec=prompt_spec,
            normal_contexts=normal_contexts,
            anomaly_contexts=anomaly_contexts,
            state_context=state_context,
            state_gate_logits=state_gate_logits,
            control_taxonomy_only=bool(args.control_taxonomy_only),
            control_taxonomy_mix_alpha=args.control_taxonomy_mix_alpha,
            generic_only_categories=generic_only_categories,
            category_to_index=category_to_index,
            scale=scale,
            pool_temperature=args.pool_temperature,
        )

        normal_mask = train_labels < 0.5
        defect_mask = ~normal_mask
        loss = outputs["anomaly_scores"].new_tensor(0.0)
        normal_bce = outputs["anomaly_scores"].new_tensor(0.0)
        defect_bce = outputs["anomaly_scores"].new_tensor(0.0)
        normal_margin = outputs["anomaly_scores"].new_tensor(0.0)
        defect_margin = outputs["anomaly_scores"].new_tensor(0.0)
        control_specific_overreach = outputs["anomaly_scores"].new_tensor(0.0)
        if normal_mask.any():
            normal_scores = outputs["anomaly_scores"][normal_mask]
            normal_weights = train_normal_loss_weight[normal_mask]
            normal_bce_terms = F.binary_cross_entropy_with_logits(
                normal_scores,
                torch.zeros_like(normal_scores),
                reduction="none",
            )
            normal_bce = (normal_bce_terms * normal_weights).sum() / torch.clamp(normal_weights.sum(), min=1.0)
            normal_margin_terms = F.relu(
                args.margin + outputs["anomaly_cos"][normal_mask] - outputs["normal_cos"][normal_mask]
            )
            normal_margin = (normal_margin_terms * normal_weights).sum() / torch.clamp(normal_weights.sum(), min=1.0)
            loss = loss + normal_bce + args.margin_weight * normal_margin
        if defect_mask.any():
            defect_scores = outputs["anomaly_scores"][defect_mask]
            defect_weights = train_defect_loss_weight[defect_mask]
            defect_bce_terms = F.binary_cross_entropy_with_logits(
                defect_scores,
                torch.ones_like(defect_scores),
                reduction="none",
            )
            defect_bce = (defect_bce_terms * defect_weights).sum() / torch.clamp(defect_weights.sum(), min=1.0)
            defect_margin_terms = F.relu(
                args.margin + outputs["normal_cos"][defect_mask] - outputs["target_anomaly_cos"][defect_mask]
            )
            defect_margin = (defect_margin_terms * defect_weights).sum() / torch.clamp(defect_weights.sum(), min=1.0)
            loss = loss + args.defect_bce_weight * defect_bce + args.defect_margin_weight * defect_margin
        control_mask = torch.tensor(
            [category in CONTROL_CATEGORIES for category in train_categories],
            dtype=torch.bool,
            device=device,
        )
        if (
            (args.control_specific_overreach_weight > 0 or args.control_defect_overreach_weight > 0)
            and control_mask.any()
        ):
            control_defect_mask = control_mask & defect_mask
            control_normal_mask = control_mask & normal_mask
            if control_normal_mask.any() and args.control_specific_overreach_weight > 0:
                control_specific_overreach = control_specific_overreach + (
                    args.control_specific_overreach_weight
                    * torch.mean(
                        F.relu(
                            outputs["specific_anomaly_cos"][control_normal_mask]
                            - outputs["generic_anomaly_cos"][control_normal_mask]
                            - args.control_specific_overreach_margin
                        )
                    )
                )
            if control_defect_mask.any() and args.control_defect_overreach_weight > 0:
                control_specific_overreach = control_specific_overreach + (
                    args.control_defect_overreach_weight
                    * torch.mean(
                        F.relu(
                            outputs["specific_anomaly_cos"][control_defect_mask]
                            - outputs["generic_anomaly_cos"][control_defect_mask]
                            - args.control_specific_overreach_margin
                        )
                    )
                )

        aux = outputs["anomaly_scores"].new_tensor(0.0)
        if outputs["aux_logits"]:
            aux_losses = [
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device))
                for logits, target in zip(outputs["aux_logits"], outputs["aux_targets"], strict=True)
            ]
            aux = torch.stack(aux_losses).mean()
            loss = loss + args.aux_defect_weight * aux

        normal_norm = torch.sum(normal_contexts * normal_contexts, dim=(1, 2))
        anomaly_norm = torch.sum(anomaly_contexts * anomaly_contexts, dim=(1, 2))
        context_penalty = args.context_l2 * torch.mean(category_penalty_weight * (normal_norm + anomaly_norm))
        state_penalty = outputs["anomaly_scores"].new_tensor(0.0)
        if state_context is not None:
            state_penalty = args.state_context_l2 * torch.mean(state_context * state_context)
        state_gate_penalty = outputs["anomaly_scores"].new_tensor(0.0)
        control_anomaly_anchor = outputs["anomaly_scores"].new_tensor(0.0)
        if state_gate_logits is not None:
            state_gates = torch.sigmoid(state_gate_logits)
            state_gate_penalty = args.state_gate_l2 * torch.mean(state_gates * state_gates)
            control_indices = [category_to_index[name] for name in CONTROL_CATEGORIES if name in category_to_index]
            if control_indices:
                control_gates = state_gates[control_indices]
                state_gate_penalty = state_gate_penalty + args.control_state_gate_weight * torch.mean(control_gates * control_gates)
        else:
            control_indices = [category_to_index[name] for name in CONTROL_CATEGORIES if name in category_to_index]
        if control_indices and args.control_anomaly_anchor_weight > 0:
            control_anomaly_anchor = args.control_anomaly_anchor_weight * torch.mean(
                anomaly_contexts[control_indices] * anomaly_contexts[control_indices]
            )
        loss = loss + context_penalty + state_penalty + state_gate_penalty + control_anomaly_anchor + control_specific_overreach
        if not torch.isfinite(loss):
            alerts.append({"epoch": epoch, "event": "non_finite_loss"})
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            holdout_outputs = compute_batch_outputs(
                features=holdout_features,
                categories=holdout_categories,
                defect_types=holdout_defects,
                text_encoder=text_encoder,
                prompt_spec=prompt_spec,
                normal_contexts=normal_contexts,
                anomaly_contexts=anomaly_contexts,
                state_context=state_context,
                state_gate_logits=state_gate_logits,
                control_taxonomy_only=bool(args.control_taxonomy_only),
                control_taxonomy_mix_alpha=args.control_taxonomy_mix_alpha,
                generic_only_categories=generic_only_categories,
                category_to_index=category_to_index,
                scale=scale,
                pool_temperature=args.pool_temperature,
            )
        holdout_scores = holdout_outputs["anomaly_scores"].detach().cpu().tolist()
        holdout_threshold_map = choose_thresholds_by_category(records=holdout_records, scores=holdout_scores)
        holdout_per_category, holdout_aggregate = evaluate_rows(
            records=holdout_records,
            scores=holdout_scores,
            threshold_map=holdout_threshold_map,
        )
        current_metric = selection_metric_with_category_bonus(
            aggregate=holdout_aggregate,
            per_category=holdout_per_category,
            control_weight=args.control_selection_weight,
            mid_weight=args.mid_selection_weight,
        )
        control_indices = [category_to_index[name] for name in CONTROL_CATEGORIES if name in category_to_index]
        append_jsonl(
            metrics_path,
            {
                "split_index": split_index,
                "epoch": epoch,
                "loss": float(loss.item()),
                "normal_bce": float(normal_bce.item()),
                "defect_bce": float(defect_bce.item()),
                "normal_margin": float(normal_margin.item()),
                "defect_margin": float(defect_margin.item()),
                "control_specific_overreach": float(control_specific_overreach.item()),
                "aux_defect": float(aux.item()),
                "context_penalty": float(context_penalty.item()),
                "state_penalty": float(state_penalty.item()),
                "state_gate_penalty": float(state_gate_penalty.item()),
                "control_anomaly_anchor": float(control_anomaly_anchor.item()),
                "holdout_image_auroc_mean": float(holdout_aggregate["image_auroc_mean"]),
                "holdout_image_ap_mean": float(holdout_aggregate["image_ap_mean"]),
                "holdout_balanced_accuracy": float(holdout_aggregate["balanced_accuracy"]),
                "selection_metric": float(current_metric),
                "normal_context_norm_mean": float(torch.norm(normal_contexts.detach(), dim=(1, 2)).mean().cpu().item()),
                "anomaly_context_norm_mean": float(torch.norm(anomaly_contexts.detach(), dim=(1, 2)).mean().cpu().item()),
                "state_context_norm": (
                    None if state_context is None else float(torch.norm(state_context.detach()).cpu().item())
                ),
                "state_gate_mean": (
                    None if state_gate_logits is None else float(torch.sigmoid(state_gate_logits).mean().detach().cpu().item())
                ),
                "control_state_gate_mean": (
                    None
                    if state_gate_logits is None or not control_indices
                    else float(torch.sigmoid(state_gate_logits[control_indices]).mean().detach().cpu().item())
                ),
                "control_context_norm_mean": (
                    float(
                        (
                            torch.norm(normal_contexts.detach()[control_indices], dim=(1, 2)).mean()
                            + torch.norm(anomaly_contexts.detach()[control_indices], dim=(1, 2)).mean()
                        ).cpu().item()
                    )
                    if control_indices
                    else None
                ),
            },
        )
        if current_metric > best_metric + args.min_delta:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {
                "normal_contexts": normal_contexts.detach().cpu().clone(),
                "anomaly_contexts": anomaly_contexts.detach().cpu().clone(),
                "state_context": None if state_context is None else state_context.detach().cpu().clone(),
                "state_gate_logits": None if state_gate_logits is None else state_gate_logits.detach().cpu().clone(),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    return {
        "best_state": best_state,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "alerts": alerts,
    }


def score_records(
    records: list[dict[str, object]],
    features_by_path: dict[str, torch.Tensor],
    prompt_spec: dict[str, dict[str, object]],
    pretrained: str,
    device: torch.device,
    args: argparse.Namespace,
    state: dict[str, torch.Tensor],
) -> tuple[list[float], list[dict[str, object]]]:
    sample_tokens = next(iter(prompt_spec.values()))["normal_tokens"]
    prompt_token_length = int(sample_tokens.shape[1])
    text_encoder = CLIPTextContextEncoder(
        context_length=prompt_token_length + args.n_ctx,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        embed_dim=1024,
        pretrained=pretrained,
    ).to(device)
    text_encoder.eval()
    category_names = sorted(prompt_spec.keys())
    category_to_index = {category: index for index, category in enumerate(category_names)}
    generic_only_categories = parse_category_list(args.generic_only_categories)
    pretrained_state = torch.jit.load(pretrained, map_location="cpu").float().state_dict()
    scale = torch.tensor(float(pretrained_state["logit_scale"].exp().item()), dtype=torch.float32, device=device)
    features = build_feature_matrix(records, features_by_path).to(device)
    categories = [str(record["category"]) for record in records]
    defect_types = [str(record.get("defect_type") or "") for record in records]
    with torch.no_grad():
        outputs = compute_batch_outputs(
            features=features,
            categories=categories,
            defect_types=defect_types,
            text_encoder=text_encoder,
            prompt_spec=prompt_spec,
            normal_contexts=state["normal_contexts"].to(device),
            anomaly_contexts=state["anomaly_contexts"].to(device),
            state_context=(
                None if state.get("state_context") is None else state["state_context"].to(device)
            ),
            state_gate_logits=(
                None if state.get("state_gate_logits") is None else state["state_gate_logits"].to(device)
            ),
            control_taxonomy_only=bool(args.control_taxonomy_only),
            control_taxonomy_mix_alpha=args.control_taxonomy_mix_alpha,
            generic_only_categories=generic_only_categories,
            category_to_index=category_to_index,
            scale=scale,
            pool_temperature=args.pool_temperature,
        )
    return outputs["anomaly_scores"].detach().cpu().tolist(), outputs["details"]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Official output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scope = args.scope.strip() or cache_dir.parent.name or cache_dir.name
    records, _ = load_records(cache_dir)
    train_records, eval_records = split_records(records)
    resplits = build_resplits(
        train_records=train_records,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        num_resplits=args.num_resplits,
    )
    prompt_spec = build_prompt_spec(train_records=train_records, prompt_token_length=args.prompt_token_length)

    preflight = {
        "route": "prompt-text-continuation",
        "variant": args.variant_tag,
        "go": True,
        "official_scope": scope,
        "official_output_dir": str(output_dir),
        "risks": [
            "This branch is PromptAD-like and category-specific; it does not reuse the failed shared-context v2/v3/v4 family.",
            "Prompt banks are built from support taxonomy only; checker must still verify bottle and zipper do not collapse.",
            "weak5 is screening only and cannot trigger official stage-doc writeback.",
        ],
    }
    (output_dir / "preflight_review.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    config = {
        "track": "new-branch/prompt-text",
        "variant": args.variant_tag,
        "scope": scope,
        "cache_dir": str(cache_dir),
        "pretrained": args.pretrained,
        "output_dir": str(output_dir),
        "feature_source": "clip_global",
        "prompt_contract": "category_specific_dual_contexts + explicit_normal_vs_anomaly_bank + promptad_like_margin",
        "selection_contract": "multisplit_holdout_mean(AUROC,AP,BalAcc)",
        "seed": args.seed,
        "num_resplits": args.num_resplits,
        "holdout_fraction": args.holdout_fraction,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "context_l2": args.context_l2,
        "control_regularization_scale": args.control_regularization_scale,
        "margin": args.margin,
        "margin_weight": args.margin_weight,
        "defect_bce_weight": args.defect_bce_weight,
        "defect_margin_weight": args.defect_margin_weight,
        "aux_defect_weight": args.aux_defect_weight,
        "use_state_bank": bool(args.use_state_bank),
        "state_context_l2": args.state_context_l2,
        "use_category_state_gate": bool(args.use_category_state_gate),
        "state_gate_init_logit": args.state_gate_init_logit,
        "state_gate_l2": args.state_gate_l2,
        "control_state_gate_weight": args.control_state_gate_weight,
        "control_loss_weight": args.control_loss_weight,
        "mid_defect_loss_weight": args.mid_defect_loss_weight,
        "control_taxonomy_only": bool(args.control_taxonomy_only),
        "control_taxonomy_mix_alpha": args.control_taxonomy_mix_alpha,
        "control_selection_weight": args.control_selection_weight,
        "mid_selection_weight": args.mid_selection_weight,
        "control_anomaly_anchor_weight": args.control_anomaly_anchor_weight,
        "control_specific_overreach_weight": args.control_specific_overreach_weight,
        "control_specific_overreach_margin": args.control_specific_overreach_margin,
        "control_defect_overreach_weight": args.control_defect_overreach_weight,
        "generic_only_categories": sorted(parse_category_list(args.generic_only_categories)),
        "n_ctx": args.n_ctx,
        "pool_temperature": args.pool_temperature,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    prompt_stats = {}
    for category, spec in prompt_spec.items():
        prompt_stats[category] = {
            "num_normal_prompts": len(spec["normal_prompts"]),
            "num_anomaly_prompts": len(spec["anomaly_prompts"]),
            "num_state_prompts": len(spec["state_prompts"]),
            "num_specific_defect_types": len(spec["specific_defect_prompts"]),
            "specific_defect_types": sorted(spec["specific_defect_prompts"].keys()),
        }
    (output_dir / "prompt_stats.json").write_text(json.dumps(prompt_stats, indent=2), encoding="utf-8")

    split_summary = {
        "num_query_eval": len(eval_records),
        "num_resplits": args.num_resplits,
        "resplits": [],
    }
    for resplit in resplits:
        split_summary["resplits"].append(
            {
                "split_index": resplit["split_index"],
                "split_seed": resplit["split_seed"],
                "num_support_train": len(resplit["support_train"]),
                "num_support_holdout": len(resplit["holdout"]),
            }
        )
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    clip_model = torch.jit.load(args.pretrained, map_location=device).eval()
    if device.type == "cuda":
        clip_model = clip_model.to(device)
    all_paths = sorted({str(record["path"]) for record in records})
    feature_maps = encode_clip_global_features(
        model=clip_model,
        paths=all_paths,
        image_size=args.clip_image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )

    metrics_path = output_dir / "train_metrics.jsonl"
    experiments_rows: list[dict[str, object]] = []
    per_category_rows: list[dict[str, object]] = []
    predictions_rows: list[dict[str, object]] = []

    pooled_holdout_records = sum([resplit["holdout"] for resplit in resplits], [])
    pooled_holdout_baseline_scores = [float(record["winner_image_score"]) for record in pooled_holdout_records]
    baseline_threshold_map = choose_thresholds_by_category(
        records=pooled_holdout_records,
        scores=pooled_holdout_baseline_scores,
    )
    baseline_scores = [float(record["winner_image_score"]) for record in eval_records]
    e0_per_category, e0_aggregate = evaluate_rows(
        records=eval_records,
        scores=baseline_scores,
        threshold_map=baseline_threshold_map,
    )
    e0_predictions = build_predictions_rows(
        experiment="E0",
        records=eval_records,
        scores=baseline_scores,
        details=[{} for _ in eval_records],
        threshold_map=baseline_threshold_map,
    )
    append_result_rows(
        experiments_rows=experiments_rows,
        per_category_rows=per_category_rows,
        predictions_rows=predictions_rows,
        experiment_name="E0",
        seed=args.seed,
        aggregate=e0_aggregate,
        per_category=e0_per_category,
        predictions=e0_predictions,
        selection_source="frozen_stage2_winner",
        threshold_source="per_category_pooled_multisplit_holdout_best_balanced_accuracy",
        threshold_value="per_category",
        scope=scope,
    )

    split_eval_scores: list[list[float]] = []
    pooled_holdout_scores: list[float] = []
    selection_epochs: list[str] = []
    alerts: list[dict[str, object]] = []
    resplit_history = {"feature_source": "clip_global", "variant": args.variant_tag, "splits": []}
    start_time = time.time()

    for resplit in resplits:
        split_result = train_split_model(
            split_index=int(resplit["split_index"]),
            train_records=resplit["support_train"],
            holdout_records=resplit["holdout"],
            features_by_path=feature_maps,
            prompt_spec=prompt_spec,
            pretrained=args.pretrained,
            device=device,
            args=args,
            metrics_path=metrics_path,
        )
        alerts.extend(split_result["alerts"])
        if split_result["alerts"]:
            break

        selection_epochs.append(str(split_result["best_epoch"]))
        holdout_scores_raw, _ = score_records(
            records=resplit["holdout"],
            features_by_path=feature_maps,
            prompt_spec=prompt_spec,
            pretrained=args.pretrained,
            device=device,
            args=args,
            state=split_result["best_state"],
        )
        normalization_stats = score_stats_by_category(records=resplit["holdout"], scores=holdout_scores_raw)
        holdout_scores = normalize_scores_by_category(
            records=resplit["holdout"],
            scores=holdout_scores_raw,
            stats=normalization_stats,
        )
        pooled_holdout_scores.extend(holdout_scores)

        eval_scores_raw, _ = score_records(
            records=eval_records,
            features_by_path=feature_maps,
            prompt_spec=prompt_spec,
            pretrained=args.pretrained,
            device=device,
            args=args,
            state=split_result["best_state"],
        )
        eval_scores = normalize_scores_by_category(
            records=eval_records,
            scores=eval_scores_raw,
            stats=normalization_stats,
        )
        split_eval_scores.append(eval_scores)
        resplit_history["splits"].append(
            {
                "split_index": int(resplit["split_index"]),
                "split_seed": int(resplit["split_seed"]),
                "best_epoch": int(split_result["best_epoch"]),
                "best_metric": float(split_result["best_metric"]),
                "normalization_stats": normalization_stats,
            }
        )

    runtime = {
        "variant": args.variant_tag,
        "runtime_seconds": float(time.time() - start_time),
        "alerts": alerts,
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (output_dir / "alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    (output_dir / "resplit_history.json").write_text(json.dumps(resplit_history, indent=2), encoding="utf-8")
    if alerts:
        raise RuntimeError(f"Training aborted due to alerts: {alerts}")

    threshold_map = choose_thresholds_by_category(records=pooled_holdout_records, scores=pooled_holdout_scores)
    plus_eval_scores = mean_score_lists(split_eval_scores)
    plus_per_category, plus_aggregate = evaluate_rows(
        records=eval_records,
        scores=plus_eval_scores,
        threshold_map=threshold_map,
    )
    plus_predictions = build_predictions_rows(
        experiment="promptad_ctx__clip_global",
        records=eval_records,
        scores=plus_eval_scores,
        details=[{} for _ in eval_records],
        threshold_map=threshold_map,
    )
    append_result_rows(
        experiments_rows=experiments_rows,
        per_category_rows=per_category_rows,
        predictions_rows=predictions_rows,
        experiment_name="promptad_ctx__clip_global",
        seed=args.seed,
        aggregate=plus_aggregate,
        per_category=plus_per_category,
        predictions=plus_predictions,
        selection_source="multisplit_holdout_mean_auroc_ap_balanced_accuracy",
        threshold_source="per_category_pooled_multisplit_holdout_best_balanced_accuracy",
        threshold_value="per_category",
        scope=scope,
        selection_epoch=",".join(selection_epochs),
    )

    (output_dir / "threshold_report.json").write_text(
        json.dumps({"E0": baseline_threshold_map, "promptad_ctx__clip_global": threshold_map}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "predictions.csv", predictions_rows)
    write_summary(output_dir / "summary.md", experiments_rows)


if __name__ == "__main__":
    main()
