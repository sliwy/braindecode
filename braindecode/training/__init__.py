"""
Functionality for skorch-based training.
"""


from .losses import CroppedLoss, TimeSeriesLoss
from .scoring import (CroppedTrialEpochScoring, PostEpochTrainScoring,
                      trial_preds_from_window_preds,)