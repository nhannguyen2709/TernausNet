import torch
from torch import nn

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth)

        # Dice Loss
        dice_coef = (2. * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1)

        return bce_loss + (1 - dice_coef)


def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2. * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().data[0]
