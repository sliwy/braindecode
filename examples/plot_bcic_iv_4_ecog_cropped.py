"""
Cropped Decoding on BCIC IV 2a Dataset
========================================

Building on the Trialwise decoding tutorial, we now do more data-efficient cropped decoding!

In Braindecode, there are two supported configurations created for training models: trialwise decoding and cropped decoding. We will explain this visually by comparing trialwise to cropped decoding.
"""

######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#


######################################################################
# Loading
# ~~~~~~~
#


######################################################################
# First, we load the data. In this tutorial, we use the functionality of
# braindecode to load datasets through
# `MOABB <https://github.com/NeuroTechX/moabb>`__ to load the BCI
# Competition IV 2a data.
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see `MNE Dataset
#    Tutorial <./plot_mne_dataset_example.html>`__ and `Numpy Dataset
#    Tutorial <./plot_custom_dataset_example.html>`__.
#
import sklearn

DATASET_PATH = '/home/maciej/projects/braindecode/BCICIV_4_mat'

import numpy as np

from braindecode.datasets.ecog_bci_competition import EcogBCICompetition4

subject_id = 1
dataset = EcogBCICompetition4(DATASET_PATH, subject_ids=[subject_id])
dataset = dataset.split('session')['train']

from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 200.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    # TODO: ensure that misc is not removed
    Preprocessor('pick_types', ecog=True, misc=True),
    Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
]
# Transform the data
preprocess(dataset, preprocessors)

######################################################################
# Create model and compute windowing parameters
# ---------------------------------------------
#


######################################################################
# In contrast to trialwise decoding, we first have to create the model
# before we can cut the dataset into windows. This is because we need to
# know the receptive field of the network to know how large the window
# stride should be.
#


######################################################################
# We first choose the compute/input window size that will be fed to the
# network during training This has to be larger than the networks
# receptive field size and can otherwise be chosen for computational
# efficiency (see explanations in the beginning of this tutorial). Here we
# choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.
#

input_window_samples = 1000

# ######################################################################
# # Create model
# # ------------
# #
#
#
# ######################################################################
# # Now we create the deep learning model! Braindecode comes with some
# # predefined convolutional neural network architectures for raw
# # time-domain EEG. Here, we use the shallow ConvNet model from `Deep
# # learning with convolutional neural networks for EEG decoding and
# # visualization <https://arxiv.org/abs/1703.05051>`__. These models are
# # pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
# # to use your own model, it just has to be a normal PyTorch
# # `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.
# #

import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 5
# Extract number of chans and time steps from dataset
n_chans = 62

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    final_conv_length=2,
)

new_model = torch.nn.Sequential()
for name, module_ in model.named_children():
    if "softmax" in name:
        continue
    new_model.add_module(name, module_)
model = new_model

# Send model to GPU
if cuda:
    model.cuda()

from braindecode.models.util import to_dense_prediction_model, get_output_shape
to_dense_prediction_model(model)

######################################################################
# To know the models’ receptive field, we calculate the shape of model
# output for a dummy input.
#

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
n_preds_per_input



# ######################################################################
# # Cut Compute Windows
# # ~~~~~~~~~~~~~~~~~~~
# #
#
#
# ######################################################################
# # Now we cut out compute windows, the inputs for the deep networks during
# # training. In the case of trialwise decoding, we just have to decide if
# # we want to cut out some part before and/or after the trial. For this
# # dataset, in our work, it often was beneficial to also cut out 500 ms
# # before the trial.
# #
#
from braindecode.preprocessing.windowers import create_fixed_length_windows

# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

from braindecode.datautil.windowers import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.

windows_dataset = create_fixed_length_windows(
    dataset,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    raw_targets='target',
    last_target_only=False,
    preload=True
)

# ######################################################################
# # Split dataset into train and valid
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
#
#
# ######################################################################
# # We can easily split the dataset using additional info stored in the
# # description attribute, in this case ``session`` column. We select
# # ``session_T`` for training and ``session_E`` for validation.
# #
#
from sklearn.model_selection import train_test_split
import torch

idx_train, idx_test_valid = train_test_split(np.arange(len(windows_dataset)),
                                             random_state=100,
                                             test_size=0.4,
                                             shuffle=False)
idx_valid, idx_test = train_test_split(idx_test_valid,
                                       test_size=0.75,
                                       shuffle=False)
train_set = torch.utils.data.Subset(windows_dataset, np.arange(100))
valid_set = torch.utils.data.Subset(windows_dataset, np.arange(100))
test_set = torch.utils.data.Subset(windows_dataset, np.arange(100))

######################################################################
# Training
# --------
#


######################################################################
# In difference to trialwise decoding, we now should supply
# ``cropped=True`` to the EEGClassifier, and ``CroppedLoss`` as the
# criterion, as well as ``criterion__loss_function`` as the loss function
# applied to the meaned predictions.
#


######################################################################
# .. note::
#    In this tutorial, we use some default parameters that we
#    have found to work well for motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.training import TimeSeriesLoss



from braindecode import EEGRegressor
# from braindecode.training.losses import TimeSeriesLoss
from braindecode.training.scoring import CroppedTimeSeriesEpochScoring

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

regressor = EEGRegressor(
    model,
    cropped=True,
    criterion=TimeSeriesLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ('r2_score', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                   lower_is_better=False,
                                                   on_train=True,
                                                   name='r2_score')
         )
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
regressor.fit(train_set, y=None, epochs=n_epochs)

xs_valid, ys_valid = [], []
for batch in valid_set:
    xs_valid.append(batch[0])
    ys_valid.append(batch[1])
ys_valid = np.stack(ys_valid)
xs_valid = np.stack(xs_valid)

xs_test, ys_test = [], []
for batch in test_set:
    xs_test.append(batch[0])
    ys_test.append(batch[1])
ys_test = np.stack(ys_test)
xs_test = np.stack(xs_test)

preds_valid = regressor.predict(xs_valid)
preds_test = regressor.predict(xs_test)

import matplotlib.pyplot as plt

plt.plot(np.arange(100)/25, preds_test[-100:, 1])
plt.plot(np.arange(100)/25, ys_test[-100:, 1])

import numpy as np

# Window correlation coefficient score
for i in range(ys_test.shape[1]):
    ys_cropped = ys_test[:, i, -preds_test.shape[2]:]
    mask = ~np.isnan(ys_cropped)

    ys_masked = ys_cropped[mask]
    preds_masked = preds_test[:, i, :][mask]
    print(preds_masked.shape)
    print(np.corrcoef(preds_masked, ys_masked)[0, 1])

# Trialwise
