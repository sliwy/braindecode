# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from numpy.ma import masked_all
from torch import nn


class CroppedLoss(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)

class TimeSeriesLoss(nn.Module):
    """Compute Loss between timeseries targets and predictions.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)
    Assumes targets are in shape:
    n_batch size x n_classes x window_len (in time)
    If the targets contain NaNs, the NaNs will be masked out and the loss will be only computed for
    predictions valid corresponding to valid target values."""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        num_preds = preds.shape[-1]
        # slice the targets to fit preds shape
        targets = targets[:, :, -num_preds:]
        # create valid targets mask
        mask = ~torch.isnan(targets)
        # select valid targets that have a matching predictions
        masked_targets = targets[mask]
        masked_preds = preds[mask]
        return self.loss_function(masked_preds, masked_targets)


