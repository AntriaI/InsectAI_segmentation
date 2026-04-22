import torch
import torch.nn as nn

"""
Recommendation for Insect segmentation task: Binary Cross Entropy + Dice Loss
Because it gives both:
1. pixel-level learning from BCE
2. mask overlap optimization from Dice
That is usually a strong default for binary segmentation.
"""

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs:  logits, shape [B, 1, H, W]
        targets: binary masks, shape [B, 1, H, W]
        """
        targets = targets.float()

        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten per image
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        probs_sum = probs.sum(dim=1)
        targets_sum = targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (probs_sum + targets_sum + self.smooth)

        # Dice loss = 1 - Dice score
        loss = 1 - dice

        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# main function to get loss function based on config
def get_loss(loss):

    if loss == "DiceLoss":
        return DiceLoss()

    elif loss == "BCEDiceLoss":
        return BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    else:
        raise ValueError(f"Loss function {loss} not supported.")
