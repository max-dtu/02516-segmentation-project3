import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true)


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with positive class weight to counter class imbalance."""

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(
            y_pred, y_true, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """Numerically stable binary Focal Loss on raw logits (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        probs = torch.sigmoid(y_pred)
        pt = probs * y_true + (1.0 - probs) * (1.0 - y_true)
        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        probs = torch.sigmoid(y_pred)
        intersection = (probs * y_true).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class PointBCELoss(nn.Module):
    """Partial binary cross-entropy loss evaluated only on clicked pixels."""

    def forward(self, y_pred, click_target, click_mask):
        bce = F.binary_cross_entropy_with_logits(y_pred, click_target, reduction="none")
        masked = bce * click_mask
        return masked.sum() / click_mask.sum().clamp(min=1.0)
