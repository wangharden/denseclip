import torch
from torch import nn


class AnomalyHead(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class ImagePoolHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ImageResidualCalibrator(nn.Module):
    def __init__(self, in_features: int, gate_bias: float = -4.0) -> None:
        super().__init__()
        self.delta_head = nn.Linear(in_features, 1)
        self.gate_head = nn.Linear(in_features, 1)
        self.reset_parameters(gate_bias=gate_bias)

    def reset_parameters(self, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.gate_head.weight)
            if self.delta_head.bias is not None:
                nn.init.zeros_(self.delta_head.bias)
            if self.gate_head.bias is not None:
                nn.init.zeros_(self.gate_head.bias)
                self.gate_head.bias.fill_(gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        frozen_logits: torch.Tensor,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.delta_head(x).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(x).squeeze(-1))
        residual = gate * delta
        final_logits = frozen_logits + residual
        if return_dict:
            return {
                "frozen_logits": frozen_logits,
                "delta": delta,
                "gate": gate,
                "residual": residual,
                "final_logits": final_logits,
            }
        return final_logits, residual, gate


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        _make_group_norm(out_channels),
        nn.ReLU(inplace=True),
    )


class MapFusionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        image_hidden_channels: int = 64,
        dropout: float = 0.1,
        topk_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.topk_ratio = topk_ratio

        self.stem = nn.Sequential(
            _conv_block(in_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
        )
        self.pixel_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.image_head = nn.Sequential(
            nn.Linear(hidden_channels * 3, image_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_channels, 1),
        )

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels = features.shape[:2]
        flat = features.reshape(batch_size, channels, -1)
        mean_pool = flat.mean(dim=-1)
        max_pool = flat.amax(dim=-1)
        topk_count = max(1, int(flat.size(-1) * self.topk_ratio))
        topk_pool = flat.topk(k=topk_count, dim=-1).values.mean(dim=-1)
        return torch.cat([mean_pool, max_pool, topk_pool], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_dict: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | dict[str, torch.Tensor]
    ):
        features = self.stem(x)
        pixel_logits = self.pixel_head(features)
        image_logits = self.image_head(self._pool_features(features)).squeeze(-1)
        if return_dict:
            outputs = {"pixel_logits": pixel_logits, "image_logits": image_logits}
            if return_features:
                outputs["features"] = features
            return outputs
        if return_features:
            return pixel_logits, image_logits, features
        return pixel_logits, image_logits


__all__ = ["AnomalyHead", "ImagePoolHead", "ImageResidualCalibrator", "MapFusionHead"]
