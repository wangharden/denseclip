from typing import Any

import torch
from torch import nn

from .backbone import DenseClipVisualEncoder
from .feature_bank import PrototypeBank
from .head import AnomalyHead
from .scoring import compute_similarity_maps, logits_to_score_outputs


class LearnedHeadAnomalyModel(nn.Module):
    def __init__(
        self,
        pretrained: str | None,
        image_size: int = 320,
        hidden_channels: int = 16,
        topk_ratio: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.topk_ratio = topk_ratio
        self.encoder = DenseClipVisualEncoder(pretrained=pretrained, input_resolution=image_size, freeze=True)
        self.head = AnomalyHead(in_channels=3, hidden_channels=hidden_channels, dropout=dropout)
        self.register_buffer("normal_prototype", torch.empty(0))
        self.register_buffer("defect_prototype", torch.empty(0))

    def set_prototype_bank(self, bank: PrototypeBank) -> None:
        self.normal_prototype = bank.normal_prototype.detach().clone()
        if bank.defect_prototype is None:
            self.defect_prototype = torch.empty(0, device=self.normal_prototype.device)
        else:
            self.defect_prototype = bank.defect_prototype.detach().clone()

    def has_prototypes(self) -> bool:
        return self.normal_prototype.numel() > 0

    def _build_similarity_tensor(self, local_features: torch.Tensor) -> torch.Tensor:
        similarity_maps = compute_similarity_maps(
            local_features=local_features,
            normal_prototype=self.normal_prototype,
            defect_prototype=self.defect_prototype if self.defect_prototype.numel() > 0 else None,
        )
        normal_map = similarity_maps["normal_map"]
        defect_map = similarity_maps["defect_map"]
        contrast_map = similarity_maps["contrast_map"]
        return torch.cat([normal_map, defect_map, contrast_map], dim=1)

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if not self.has_prototypes():
            raise RuntimeError("Prototype bank has not been set.")

        with torch.no_grad():
            encoded = self.encoder(images)
            local_features = encoded["local"]

        similarity = self._build_similarity_tensor(local_features)
        patch_logits = self.head(similarity)
        scored = logits_to_score_outputs(
            score_logits=patch_logits,
            image_size=self.image_size,
            topk_ratio=self.topk_ratio,
        )
        return {
            "similarity": similarity,
            "patch_logits": patch_logits,
            "upsampled_logits": scored["upsampled_logits"],
            "image_logits": scored["image_logits"],
            "upsampled_scores": scored["upsampled_scores"],
            "image_scores": scored["image_scores"],
        }
