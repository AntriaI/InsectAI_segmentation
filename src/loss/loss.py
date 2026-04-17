import torch
import torch.nn as nn
from .focalloss import FocalLoss

#main function to get loss function based on config
def get_loss(loss):

    if loss == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()

    elif loss == "FocalLoss":
        return FocalLoss(alpha=0.25, gamma=2.0)

    else:
        raise ValueError(f"Loss function {loss} not supported.")



if __name__ == "__main__":
    import torch

    logits = torch.randn(3, 1)
    targets = torch.tensor([[1.0], [0.0], [1.0]])

    loss_fn = get_loss("FocalLoss")

    loss = loss_fn(logits, targets)

    print("Loss:", loss.item())