import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_ROOT = REPO_ROOT / "detection" / "denseclip"
if str(TOKENIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_ROOT))

from untils import tokenize

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


CONTROL_CATEGORIES = ("bottle", "zipper")
DEFAULT_NORMAL_TEMPLATES = (
    "a photo of a normal {category}.",
    "a photo of a good {category}.",
    "a photo of a flawless {category}.",
)
DEFAULT_GENERIC_DEFECT_TEMPLATES = (
    "a photo of a defective {category}.",
    "a damaged {category}.",
    "a photo of an anomalous {category}.",
)
DEFAULT_SPECIFIC_DEFECT_TEMPLATES = (
    "a photo of a {category} with {defect}.",
    "a defective {category} showing {defect}.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PromptAD-like learnable context prompts on frozen CLIP image features."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_ctx_v1_cooplike_screen",
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
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--context-l2", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--aux-defect-weight", type=float, default=0.25)
    parser.add_argument("--pool-temperature", type=float, default=0.10)
    parser.add_argument("--prompt-token-length", type=int, default=24)
    parser.add_argument("--n-ctx", type=int, default=8)
    parser.add_argument("--variant-tag", default="prompt_ctx_v2_gated")
    parser.add_argument("--gate-init-logit", type=float, default=-1.5)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    parser.add_argument("--use-category-defect-residual", action="store_true")
    parser.add_argument("--defect-category-delta-l2", type=float, default=1e-4)
    parser.add_argument("--use-category-normal-residual", action="store_true")
    parser.add_argument("--normal-category-delta-l2", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_defect_name(defect_type: str) -> str:
    return defect_type.replace("_", " ").replace("-", " ").strip()


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class CLIPTextContextEncoder(nn.Module):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        embed_dim: int,
        pretrained: str,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.init_from_pretrained(pretrained)
        for param in self.parameters():
            param.requires_grad = False

    def init_from_pretrained(self, pretrained: str) -> None:
        checkpoint = torch.jit.load(pretrained, map_location="cpu").float().state_dict()
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            if key.startswith("transformer."):
                state_dict[key] = value
            if key == "positional_embedding" or key == "text_projection" or key.startswith("token_embedding") or key.startswith("ln_final"):
                if key == "positional_embedding" and value.size(0) > self.context_length:
                    value = value[: self.context_length]
                state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)

    def build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x_text = self.token_embedding(text)
        num_classes, base_len, width = x_text.shape
        batch_size, n_ctx, _ = context.shape
        eos_index = text.argmax(dim=-1) + n_ctx
        eos_index = eos_index.reshape(1, num_classes).expand(batch_size, num_classes).reshape(-1)

        x_text = x_text.reshape(1, num_classes, base_len, width).expand(batch_size, num_classes, base_len, width)
        expanded_context = context.reshape(batch_size, 1, n_ctx, width).expand(batch_size, num_classes, n_ctx, width)
        x = torch.cat([x_text[:, :, 0:1], expanded_context, x_text[:, :, 1:]], dim=2).reshape(
            batch_size * num_classes,
            base_len + n_ctx,
            width,
        )
        x = x + self.positional_embedding[: base_len + n_ctx]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_index] @ self.text_projection
        return x.reshape(batch_size, num_classes, self.embed_dim)


def build_prompt_spec(
    records: list[dict[str, object]],
    prompt_token_length: int,
) -> dict[str, dict[str, object]]:
    defect_types_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        category = str(record["category"])
        defect_type = str(record.get("defect_type") or "").strip()
        if defect_type and defect_type != "good":
            defect_types_by_category[category].add(defect_type)

    prompt_spec: dict[str, dict[str, object]] = {}
    for category in sorted({str(record["category"]) for record in records}):
        normal_prompts = [template.format(category=category) for template in DEFAULT_NORMAL_TEMPLATES]
        generic_defect_prompts = [
            template.format(category=category) for template in DEFAULT_GENERIC_DEFECT_TEMPLATES
        ]
        specific_prompts: dict[str, list[str]] = {}
        for defect_type in sorted(defect_types_by_category.get(category, set())):
            defect_name = sanitize_defect_name(defect_type)
            specific_prompts[defect_type] = [
                template.format(category=category, defect=defect_name) for template in DEFAULT_SPECIFIC_DEFECT_TEMPLATES
            ]
        prompt_spec[category] = {
            "normal_prompts": normal_prompts,
            "generic_defect_prompts": generic_defect_prompts,
            "specific_defect_prompts": specific_prompts,
            "normal_tokens": tokenize(normal_prompts, context_length=prompt_token_length, truncate=True),
            "generic_defect_tokens": tokenize(
                generic_defect_prompts,
                context_length=prompt_token_length,
                truncate=True,
            ),
            "specific_defect_tokens": {
                defect_type: tokenize(prompts, context_length=prompt_token_length, truncate=True)
                for defect_type, prompts in specific_prompts.items()
            },
        }
    return prompt_spec


def category_embed_average(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, dim=-1)
    averaged = normalized.mean(dim=0, keepdim=True)
    return F.normalize(averaged, dim=-1)[0]


def encode_category_bank(
    category: str,
    prompt_spec: dict[str, dict[str, object]],
    text_encoder: CLIPTextContextEncoder,
    normal_context: torch.Tensor,
    normal_context_delta: torch.Tensor | None,
    defect_context: torch.Tensor,
    defect_context_delta: torch.Tensor | None,
    category_gate_logit: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    spec = prompt_spec[category]
    gate = torch.sigmoid(category_gate_logit)
    category_normal_context = normal_context
    if normal_context_delta is not None:
        category_normal_context = category_normal_context + normal_context_delta
    gated_normal_context = category_normal_context * gate
    category_defect_context = defect_context
    if defect_context_delta is not None:
        category_defect_context = category_defect_context + defect_context_delta
    gated_defect_context = category_defect_context * gate
    normal_tokens = spec["normal_tokens"].to(device)
    normal_embeddings = text_encoder(normal_tokens, gated_normal_context.unsqueeze(0))[0]
    normal_prototype = category_embed_average(normal_embeddings)

    generic_tokens = spec["generic_defect_tokens"].to(device)
    generic_embeddings = text_encoder(generic_tokens, gated_defect_context.unsqueeze(0))[0]
    generic_prototype = category_embed_average(generic_embeddings)

    specific_names: list[str] = []
    specific_prototypes: list[torch.Tensor] = []
    for defect_type, tokens in spec["specific_defect_tokens"].items():
        prompt_embeddings = text_encoder(tokens.to(device), gated_defect_context.unsqueeze(0))[0]
        specific_names.append(defect_type)
        specific_prototypes.append(category_embed_average(prompt_embeddings))

    if specific_prototypes:
        specific_tensor = torch.stack(specific_prototypes, dim=0)
        all_defect_embeddings = torch.cat([generic_prototype.unsqueeze(0), specific_tensor], dim=0)
        all_defect_names = ["generic_anomaly", *specific_names]
    else:
        specific_tensor = torch.empty((0, generic_prototype.shape[0]), device=device)
        all_defect_embeddings = generic_prototype.unsqueeze(0)
        all_defect_names = ["generic_anomaly"]

    return {
        "normal_embedding": normal_prototype,
        "generic_defect_embedding": generic_prototype,
        "specific_defect_embeddings": specific_tensor,
        "specific_defect_names": specific_names,
        "all_defect_embeddings": all_defect_embeddings,
        "all_defect_names": all_defect_names,
        "gate": gate,
    }


def pooled_defect_similarity(defect_cos: torch.Tensor, pool_temperature: float) -> torch.Tensor:
    return torch.logsumexp(defect_cos / pool_temperature, dim=0) * pool_temperature


def compute_batch_outputs(
    features: torch.Tensor,
    categories: list[str],
    defect_types: list[str],
    text_encoder: CLIPTextContextEncoder,
    prompt_spec: dict[str, dict[str, object]],
    normal_context: torch.Tensor,
    normal_context_deltas: torch.Tensor | None,
    defect_context: torch.Tensor,
    defect_context_deltas: torch.Tensor | None,
    category_gate_logits: torch.Tensor,
    category_to_index: dict[str, int],
    logit_scale_log: torch.Tensor,
    pool_temperature: float,
) -> dict[str, object]:
    scale = torch.clamp(logit_scale_log.exp(), min=1.0, max=100.0)
    cache: dict[str, dict[str, object]] = {}
    anomaly_scores: list[torch.Tensor] = []
    normal_cos_values: list[torch.Tensor] = []
    pooled_defect_cos_values: list[torch.Tensor] = []
    target_defect_cos_values: list[torch.Tensor] = []
    aux_logits: list[torch.Tensor] = []
    aux_targets: list[int] = []
    details: list[dict[str, object]] = []

    for feature, category, defect_type in zip(features, categories, defect_types, strict=True):
        if category not in cache:
            cache[category] = encode_category_bank(
                category=category,
                prompt_spec=prompt_spec,
                text_encoder=text_encoder,
                normal_context=normal_context,
                normal_context_delta=(
                    None
                    if normal_context_deltas is None
                    else normal_context_deltas[category_to_index[category]]
                ),
                defect_context=defect_context,
                defect_context_delta=(
                    None
                    if defect_context_deltas is None
                    else defect_context_deltas[category_to_index[category]]
                ),
                category_gate_logit=category_gate_logits[category_to_index[category]],
                device=feature.device,
            )
        bank = cache[category]
        feature = F.normalize(feature.unsqueeze(0), dim=-1)[0]
        normal_cos = torch.dot(feature, bank["normal_embedding"])
        defect_cos = torch.matmul(bank["all_defect_embeddings"], feature)
        pooled_defect_cos = pooled_defect_similarity(defect_cos=defect_cos, pool_temperature=pool_temperature)
        anomaly_scores.append(scale * (pooled_defect_cos - normal_cos))
        normal_cos_values.append(normal_cos)
        pooled_defect_cos_values.append(pooled_defect_cos)

        target_defect_cos = pooled_defect_cos
        specific_names: list[str] = bank["specific_defect_names"]
        if defect_type and defect_type != "good" and specific_names:
            try:
                target_index = specific_names.index(defect_type)
            except ValueError:
                target_index = -1
            if target_index >= 0:
                specific_cos = torch.matmul(bank["specific_defect_embeddings"], feature)
                aux_logits.append(scale * specific_cos)
                aux_targets.append(int(target_index))
                target_specific = specific_cos[target_index]
                target_defect_cos = torch.maximum(target_specific, defect_cos[0])
        target_defect_cos_values.append(target_defect_cos)

        max_defect_value, max_defect_index = torch.max(defect_cos, dim=0)
        details.append(
            {
                "normal_score": float((scale * normal_cos).detach().cpu().item()),
                "max_defect_score": float((scale * max_defect_value).detach().cpu().item()),
                "top1_prompt": bank["all_defect_names"][int(max_defect_index.detach().cpu().item())],
                "top1_prompt_score": float((scale * max_defect_value).detach().cpu().item()),
                "category_gate": float(bank["gate"].detach().cpu().item()),
            }
        )

    return {
        "anomaly_scores": torch.stack(anomaly_scores, dim=0),
        "normal_cos": torch.stack(normal_cos_values, dim=0),
        "pooled_defect_cos": torch.stack(pooled_defect_cos_values, dim=0),
        "target_defect_cos": torch.stack(target_defect_cos_values, dim=0),
        "aux_logits": aux_logits,
        "aux_targets": aux_targets,
        "details": details,
        "scale": scale,
    }


def selection_metric(aggregate: dict[str, float]) -> float:
    return float(
        (
            float(aggregate["image_auroc_mean"])
            + float(aggregate["image_ap_mean"])
            + float(aggregate["balanced_accuracy"])
        )
        / 3.0
    )


def score_stats_by_category(records: list[dict[str, object]], scores: list[float]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        grouped[str(record["category"])].append(float(score))
    stats: dict[str, dict[str, float]] = {}
    for category, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std())
        stats[category] = {
            "mean": float(array.mean()),
            "std": std if std > 1e-6 else 1.0,
        }
    return stats


def normalize_scores_by_category(
    records: list[dict[str, object]],
    scores: list[float],
    stats: dict[str, dict[str, float]],
) -> list[float]:
    normalized: list[float] = []
    for record, score in zip(records, scores, strict=True):
        category = str(record["category"])
        mean = float(stats[category]["mean"])
        std = float(stats[category]["std"])
        normalized.append((float(score) - mean) / std)
    return normalized


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

    pretrained_state = torch.jit.load(pretrained, map_location="cpu").float().state_dict()
    category_names = sorted(prompt_spec.keys())
    category_to_index = {category: index for index, category in enumerate(category_names)}
    initial_logit_scale = float(pretrained_state["logit_scale"].exp().item())
    normal_context = nn.Parameter(torch.zeros(args.n_ctx, 512, device=device))
    normal_context_deltas: nn.Parameter | None = None
    if args.use_category_normal_residual:
        normal_context_deltas = nn.Parameter(
            torch.zeros(len(category_names), args.n_ctx, 512, device=device)
        )
    defect_context = nn.Parameter(torch.zeros(args.n_ctx, 512, device=device))
    category_gate_logits = nn.Parameter(
        torch.full((len(category_names),), float(args.gate_init_logit), device=device)
    )
    defect_context_deltas: nn.Parameter | None = None
    if args.use_category_defect_residual:
        defect_context_deltas = nn.Parameter(
            torch.zeros(len(category_names), args.n_ctx, 512, device=device)
        )
    logit_scale_log = torch.tensor(math.log(initial_logit_scale), dtype=torch.float32, device=device)
    nn.init.normal_(normal_context, std=0.02)
    nn.init.normal_(defect_context, std=0.02)

    optimizer_params: list[torch.Tensor] = [normal_context, defect_context, category_gate_logits]
    if normal_context_deltas is not None:
        optimizer_params.append(normal_context_deltas)
    if defect_context_deltas is not None:
        optimizer_params.append(defect_context_deltas)
    optimizer = AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_features = build_feature_matrix(train_records, features_by_path)
    train_labels = torch.tensor([float(record["label"]) for record in train_records], dtype=torch.float32)
    holdout_features = build_feature_matrix(holdout_records, features_by_path)

    train_categories = [str(record["category"]) for record in train_records]
    train_defects = [str(record.get("defect_type") or "") for record in train_records]
    holdout_categories = [str(record["category"]) for record in holdout_records]
    holdout_defects = [str(record.get("defect_type") or "") for record in holdout_records]

    best_state = {
        "normal_context": normal_context.detach().cpu().clone(),
        "normal_context_deltas": None if normal_context_deltas is None else normal_context_deltas.detach().cpu().clone(),
        "defect_context": defect_context.detach().cpu().clone(),
        "defect_context_deltas": None if defect_context_deltas is None else defect_context_deltas.detach().cpu().clone(),
        "category_gate_logits": category_gate_logits.detach().cpu().clone(),
        "logit_scale_log": logit_scale_log.detach().cpu().clone(),
    }
    best_epoch = 0
    best_metric = float("-inf")
    stale_epochs = 0
    alerts: list[dict[str, object]] = []
    train_features_device = train_features.to(device)
    train_labels_device = train_labels.to(device)
    holdout_features_device = holdout_features.to(device)

    for epoch in range(1, args.epochs + 1):
        outputs = compute_batch_outputs(
            features=train_features_device,
            categories=train_categories,
            defect_types=train_defects,
            text_encoder=text_encoder,
            prompt_spec=prompt_spec,
            normal_context=normal_context,
            normal_context_deltas=normal_context_deltas,
            defect_context=defect_context,
            defect_context_deltas=defect_context_deltas,
            category_gate_logits=category_gate_logits,
            category_to_index=category_to_index,
            logit_scale_log=logit_scale_log,
            pool_temperature=args.pool_temperature,
        )
        anomaly_scores = outputs["anomaly_scores"]
        bce = F.binary_cross_entropy_with_logits(anomaly_scores, train_labels_device)
        margin_normal = F.relu(args.margin + outputs["pooled_defect_cos"] - outputs["normal_cos"])
        margin_defect = F.relu(args.margin + outputs["normal_cos"] - outputs["target_defect_cos"])
        margin = ((1.0 - train_labels_device) * margin_normal + train_labels_device * margin_defect).mean()
        aux = anomaly_scores.new_tensor(0.0)
        if outputs["aux_logits"]:
            aux_losses = [
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device))
                for logits, target in zip(outputs["aux_logits"], outputs["aux_targets"], strict=True)
            ]
            aux = torch.stack(aux_losses).mean()
        l2 = args.context_l2 * (
            torch.sum(normal_context * normal_context) + torch.sum(defect_context * defect_context)
        )
        normal_delta_penalty = anomaly_scores.new_tensor(0.0)
        if normal_context_deltas is not None:
            normal_delta_penalty = args.normal_category_delta_l2 * torch.mean(normal_context_deltas * normal_context_deltas)
        defect_delta_penalty = anomaly_scores.new_tensor(0.0)
        if defect_context_deltas is not None:
            defect_delta_penalty = args.defect_category_delta_l2 * torch.mean(defect_context_deltas * defect_context_deltas)
        gate_penalty = args.gate_l2 * torch.mean(torch.sigmoid(category_gate_logits) ** 2)
        loss = (
            bce
            + args.margin_weight * margin
            + args.aux_defect_weight * aux
            + l2
            + gate_penalty
            + normal_delta_penalty
            + defect_delta_penalty
        )
        if not torch.isfinite(loss):
            alerts.append({"epoch": epoch, "event": "non_finite_loss"})
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            holdout_outputs = compute_batch_outputs(
                features=holdout_features_device,
                categories=holdout_categories,
                defect_types=holdout_defects,
                text_encoder=text_encoder,
                prompt_spec=prompt_spec,
                normal_context=normal_context,
                normal_context_deltas=normal_context_deltas,
                defect_context=defect_context,
                defect_context_deltas=defect_context_deltas,
                category_gate_logits=category_gate_logits,
                category_to_index=category_to_index,
                logit_scale_log=logit_scale_log,
                pool_temperature=args.pool_temperature,
            )
        holdout_scores = holdout_outputs["anomaly_scores"].detach().cpu().tolist()
        holdout_threshold_map = choose_thresholds_by_category(records=holdout_records, scores=holdout_scores)
        _, holdout_aggregate = evaluate_rows(
            records=holdout_records,
            scores=holdout_scores,
            threshold_map=holdout_threshold_map,
        )
        current_metric = selection_metric(holdout_aggregate)
        append_jsonl(
            metrics_path,
            {
                "split_index": split_index,
                "epoch": epoch,
                "loss": float(loss.item()),
                "bce": float(bce.item()),
                "margin": float(margin.item()),
                "aux_defect": float(aux.item()),
                "gate_penalty": float(gate_penalty.item()),
                "normal_delta_penalty": float(normal_delta_penalty.item()),
                "defect_delta_penalty": float(defect_delta_penalty.item()),
                "holdout_image_auroc_mean": float(holdout_aggregate["image_auroc_mean"]),
                "holdout_image_ap_mean": float(holdout_aggregate["image_ap_mean"]),
                "holdout_balanced_accuracy": float(holdout_aggregate["balanced_accuracy"]),
                "selection_metric": current_metric,
                "logit_scale": float(holdout_outputs["scale"].detach().cpu().item()),
                "gate_mean": float(torch.sigmoid(category_gate_logits).mean().detach().cpu().item()),
                "gate_control_mean": float(
                    torch.sigmoid(
                        torch.stack(
                            [
                                category_gate_logits[category_to_index[name]]
                                for name in CONTROL_CATEGORIES
                                if name in category_to_index
                            ]
                        )
                    ).mean().detach().cpu().item()
                ) if any(name in category_to_index for name in CONTROL_CATEGORIES) else None,
                "defect_delta_norm_mean": float(
                    0.0
                    if defect_context_deltas is None
                    else torch.norm(defect_context_deltas.detach(), dim=(1, 2)).mean().cpu().item()
                ),
                "normal_delta_norm_mean": float(
                    0.0
                    if normal_context_deltas is None
                    else torch.norm(normal_context_deltas.detach(), dim=(1, 2)).mean().cpu().item()
                ),
                "normal_context_norm": float(torch.norm(normal_context.detach()).cpu().item()),
                "defect_context_norm": float(torch.norm(defect_context.detach()).cpu().item()),
            },
        )
        if current_metric > best_metric + args.min_delta:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {
                "normal_context": normal_context.detach().cpu().clone(),
                "normal_context_deltas": (
                    None if normal_context_deltas is None else normal_context_deltas.detach().cpu().clone()
                ),
                "defect_context": defect_context.detach().cpu().clone(),
                "defect_context_deltas": (
                    None if defect_context_deltas is None else defect_context_deltas.detach().cpu().clone()
                ),
                "category_gate_logits": category_gate_logits.detach().cpu().clone(),
                "logit_scale_log": logit_scale_log.detach().cpu().clone(),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
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
    feature_tensor = build_feature_matrix(records, features_by_path).to(device)
    categories = [str(record["category"]) for record in records]
    defect_types = [str(record.get("defect_type") or "") for record in records]
    category_names = sorted(prompt_spec.keys())
    category_to_index = {category: index for index, category in enumerate(category_names)}
    with torch.no_grad():
        outputs = compute_batch_outputs(
            features=feature_tensor,
            categories=categories,
            defect_types=defect_types,
            text_encoder=text_encoder,
            prompt_spec=prompt_spec,
            normal_context=state["normal_context"].to(device),
            normal_context_deltas=(
                None
                if state.get("normal_context_deltas") is None
                else state["normal_context_deltas"].to(device)
            ),
            defect_context=state["defect_context"].to(device),
            defect_context_deltas=(
                None
                if state.get("defect_context_deltas") is None
                else state["defect_context_deltas"].to(device)
            ),
            category_gate_logits=state["category_gate_logits"].to(device),
            category_to_index=category_to_index,
            logit_scale_log=state["logit_scale_log"].to(device),
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

    preflight = {
        "route": "prompt-text-continuation",
        "variant": args.variant_tag,
        "go": True,
        "official_scope": scope,
        "official_output_dir": str(output_dir),
        "risks": [
            "Defect-type coverage in support_defect is sparse for several categories; defect-class supervision is auxiliary only.",
            "Selection metric mixes AUROC/AP/balanced_accuracy to avoid AUROC-only winner drift, but holdout remains small.",
            "Only clip_global is retained; denseclip_global and layer3_gap stay out of mainline based on prior closeouts.",
            "Category gates must stay conservative; if bottle/zipper still regress materially, the candidate is not retained.",
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
        "clip_image_size": args.clip_image_size,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "num_resplits": args.num_resplits,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "context_l2": args.context_l2,
        "margin": args.margin,
        "margin_weight": args.margin_weight,
        "aux_defect_weight": args.aux_defect_weight,
        "pool_temperature": args.pool_temperature,
        "prompt_token_length": args.prompt_token_length,
        "n_ctx": args.n_ctx,
        "gate_init_logit": args.gate_init_logit,
        "gate_l2": args.gate_l2,
        "use_category_normal_residual": bool(args.use_category_normal_residual),
        "normal_category_delta_l2": args.normal_category_delta_l2,
        "use_category_defect_residual": bool(args.use_category_defect_residual),
        "defect_category_delta_l2": args.defect_category_delta_l2,
        "feature_source": "clip_global",
        "selection_contract": "multisplit_holdout_mean(AUROC,AP,BalAcc)",
        "prompt_contract": " + ".join(
            [
                "learnable_context_tokens",
                "category_scalar_gate",
                *(
                    ["category_specific_normal_residual"]
                    if args.use_category_normal_residual
                    else []
                ),
                *(
                    ["category_specific_defect_residual"]
                    if args.use_category_defect_residual
                    else []
                ),
                "generic_defect",
                "specific_defect_aux",
            ]
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    split_summary = {
        "num_query_eval": len(eval_records),
        "num_resplits": args.num_resplits,
        "resplits": [],
    }
    for resplit in resplits:
        split_entry = {
            "split_index": resplit["split_index"],
            "split_seed": resplit["split_seed"],
            "num_support_train": len(resplit["support_train"]),
            "num_support_holdout": len(resplit["holdout"]),
            "support_train_by_category_label": {},
            "support_holdout_by_category_label": {},
        }
        for key_name, subset_name in (
            ("support_train_by_category_label", "support_train"),
            ("support_holdout_by_category_label", "holdout"),
        ):
            counts: dict[str, int] = defaultdict(int)
            for record in resplit[subset_name]:
                counts[f"{record['category']}|label={record['label']}"] += 1
            split_entry[key_name] = dict(sorted(counts.items()))
        split_summary["resplits"].append(split_entry)
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    prompt_spec = build_prompt_spec(records=records, prompt_token_length=args.prompt_token_length)
    prompt_stats = {}
    for category, spec in prompt_spec.items():
        prompt_stats[category] = {
            "num_normal_prompts": len(spec["normal_prompts"]),
            "num_generic_defect_prompts": len(spec["generic_defect_prompts"]),
            "num_specific_defect_types": len(spec["specific_defect_prompts"]),
            "specific_defect_types": sorted(spec["specific_defect_prompts"].keys()),
        }
    (output_dir / "prompt_stats.json").write_text(json.dumps(prompt_stats, indent=2), encoding="utf-8")

    clip_model = torch.jit.load(args.pretrained, map_location=device).eval()
    if device.type == "cuda":
        clip_model = clip_model.to(device)

    metrics_path = output_dir / "train_metrics.jsonl"
    all_paths = sorted({str(record["path"]) for record in records})
    feature_maps = encode_clip_global_features(
        model=clip_model,
        paths=all_paths,
        image_size=args.clip_image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )

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

    start_time = time.time()
    split_eval_scores: list[list[float]] = []
    split_eval_details: list[list[dict[str, object]]] = []
    pooled_holdout_scores: list[float] = []
    selection_epochs: list[str] = []
    resplit_history = {"feature_source": "clip_global", "variant": args.variant_tag, "splits": []}
    alerts: list[dict[str, object]] = []

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
        eval_scores_raw, eval_details = score_records(
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
        split_eval_details.append(eval_details)
        resplit_history["splits"].append(
            {
                "split_index": int(resplit["split_index"]),
                "split_seed": int(resplit["split_seed"]),
                "best_epoch": int(split_result["best_epoch"]),
                "best_metric": float(split_result["best_metric"]),
                "num_support_train": len(resplit["support_train"]),
                "num_support_holdout": len(resplit["holdout"]),
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
    representative_details = split_eval_details[0] if split_eval_details else [{} for _ in eval_records]
    plus_per_category, plus_aggregate = evaluate_rows(
        records=eval_records,
        scores=plus_eval_scores,
        threshold_map=threshold_map,
    )
    plus_predictions = build_predictions_rows(
        experiment="prompt_ctx_plus__clip_global",
        records=eval_records,
        scores=plus_eval_scores,
        details=representative_details,
        threshold_map=threshold_map,
    )
    append_result_rows(
        experiments_rows=experiments_rows,
        per_category_rows=per_category_rows,
        predictions_rows=predictions_rows,
        experiment_name="prompt_ctx_plus__clip_global",
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

    threshold_report = {
        "E0": baseline_threshold_map,
        "prompt_ctx_plus__clip_global": threshold_map,
    }
    (output_dir / "threshold_report.json").write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")

    import csv

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

    write_csv(output_dir / "experiments.csv", experiments_rows)
    write_csv(output_dir / "per_category.csv", per_category_rows)
    write_csv(output_dir / "predictions.csv", predictions_rows)
    write_summary(output_dir / "summary.md", experiments_rows)


if __name__ == "__main__":
    main()
