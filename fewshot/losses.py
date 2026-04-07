import torch
import torch.nn.functional as F


def pairwise_margin_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    smooth: bool = False,
) -> torch.Tensor:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).to(dtype=torch.bool)
    positive = scores[labels]
    negative = scores[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.new_zeros(())
    margins = margin - positive[:, None] + negative[None, :]
    if smooth:
        return F.softplus(margins).mean()
    return torch.relu(margins).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    if targets.shape != probs.shape:
        targets = targets.reshape_as(probs)

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth).clamp_min(1e-6)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
__all__ = ["pairwise_margin_rank", "dice_loss"]
