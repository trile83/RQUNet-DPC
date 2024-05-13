from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def _flatten_temporal_dim(preds, targets):
    # Flatten temporal dim if exists
    if preds.ndim > 2 and preds.ndim < 4:
        preds = rearrange(preds, "b t c -> (b t) c")
        targets = rearrange(targets, "b t -> (b t)")
    if preds.ndim > 4:
        preds = rearrange(preds, "b t c h w-> (b t) c h w")
        targets = rearrange(targets, "b t h w-> (b t) h w")
    return preds, targets



class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return dict(loss=self.loss(preds, targets))


class MultiTemporalCrossEntropy(nn.Module):
    """Cross Entropy wrapper for handling MultiTemporal inputs/outputs"""

    def __init__(self, class_weights: list = None):
        super().__init__()
        weight = torch.FloatTensor(class_weights) if class_weights else None
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Mask loss, default mask value = -100
        if mask is None:
            masked_targets = targets
        else:
            masked_targets = targets.clone()
            masked_targets[mask.to(torch.bool)] = -100  # if not a clone, then targets become less useful downstream

        preds, masked_targets = _flatten_temporal_dim(preds, masked_targets)
        return dict(loss=self.loss(preds, masked_targets))


class FocalTversky(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smoothing: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Rely on the forward method of each of the losses to handle masking and ignoring indices
        preds, targets = _flatten_temporal_dim(preds.softmax(axis=2)[:, :, 1:, ...].contiguous(), targets)

        # flatten tensors, mask if mask
        try:
            mask_indices = mask.view(-1).to(torch.bool)
        except AttributeError:
            mask_indices = torch.zeros(preds.view(-1).shape, dtype=torch.bool)

        # masked means ignore, dont compute loss on these pixels,
        # so compute this loss only where there is no mask, i.e. ~mask
        masked_preds = preds.clone().view(-1)[~mask_indices]
        masked_targets = targets.clone().view(-1)[~mask_indices]

        tp = (masked_preds * masked_targets).sum()
        fp = ((1 - masked_targets) * masked_preds).sum()
        fn = (masked_targets * (1 - masked_preds)).sum()

        tversky = (tp + self.smoothing) / (tp + self.alpha * fp + self.beta * fn + self.smoothing)
        focal_tversky = (1 - tversky) ** self.gamma

        return dict(loss=focal_tversky)


class InfoNCE(nn.Module):
    """
    Information Noise Contrastive Estimation Loss
    https://paperswithcode.com/method/infonce
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_hat: torch.Tensor, z_true: torch.Tensor) -> Dict[str, torch.Tensor]:

        bs, k, z_dim = z_true.shape
        device = z_true.device

        loss = 0.0
        acc = 0.0

        # Loop over future steps k
        for i in range(k):

            # calculate the log density ratio (basically similarity via dot product)
            # (bs, bs) where the diagonal corresponds to dot product of vectors from same sequence in batch
            logits = torch.mm(z_true[:, i, :], z_hat[:, i, :].t())

            # Note positive samples are the diagonal components. Need to maximize these in loss
            # Negative samples are samples from other sequences in the batch
            labels = torch.arange(bs, device=device)
            loss += F.cross_entropy(logits, labels)

            # Correct if highest prob is the diagonal
            y_pred = torch.argmax(F.softmax(logits, dim=0), dim=0)
            acc += torch.sum(torch.eq(y_pred, labels))

        loss /= k
        acc /= bs * k

        return dict(loss=loss, acc=acc)
