from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("pool", nn.AvgPool2d(stride)),
                        ("conv", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                        ("bn", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape
        x = x.reshape(batch_size, x.shape[1], height * width).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.spacial_dim, self.embed_dim)[:height, :width]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        local_feat = x[:, :, 1:].reshape(batch_size, -1, height, width)
        return global_feat, local_feat


class CLIPResNetWithAttention(nn.Module):
    def __init__(
        self,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
        output_dim: int = 1024,
        input_resolution: int = 224,
        width: int = 64,
        pretrained: str | None = None,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)

    def init_weights(self, pretrained: str | None = None) -> None:
        pretrained = pretrained or self.pretrained
        if not pretrained:
            return
        checkpoint = torch.jit.load(pretrained, map_location="cpu").float().state_dict()
        state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith("visual."):
                continue
            new_key = key.replace("visual.", "")
            if "positional_embedding" in new_key and self.attnpool.positional_embedding.shape != value.shape:
                cls_pos = value[0:1, :]
                grid_h = grid_w = self.input_resolution // 32
                spatial_pos = F.interpolate(
                    value[1:].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2),
                    size=(grid_h, grid_w),
                    mode="bilinear",
                    align_corners=False,
                )
                spatial_pos = spatial_pos.reshape(cls_pos.shape[1], grid_h * grid_w).permute(1, 0)
                value = torch.cat([cls_pos, spatial_pos], dim=0)
            state_dict[new_key] = value
        self.load_state_dict(state_dict, strict=False)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = x.type(self.conv1.weight.dtype)
        for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)):
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        global_feat, local_feat = self.attnpool(out4)
        return out1, out2, out3, out4, global_feat, local_feat


class DenseClipVisualEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str | None,
        input_resolution: int = 320,
        output_dim: int = 1024,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.visual = CLIPResNetWithAttention(
            layers=layers,
            output_dim=output_dim,
            input_resolution=input_resolution,
            pretrained=pretrained,
        )
        self.visual.init_weights(pretrained)
        if freeze:
            self.requires_grad_(False)
            self.visual.eval()

    def train(self, mode: bool = True) -> "DenseClipVisualEncoder":
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out1, out2, out3, out4, global_feat, local_feat = self.visual(x)
        return {
            "layer1": out1,
            "layer2": out2,
            "layer3": out3,
            "layer4": out4,
            "global": global_feat,
            "local": local_feat,
        }
