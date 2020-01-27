from collections import OrderedDict
from datetime import datetime

import numpy as np
import os.path
import torch
from braindecode.classifier import EEGClassifier
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.models.deep4 import Deep4Net
from sklearn.metrics import accuracy_score
from skorch.callbacks import Checkpoint, EpochScoring, ProgressBar
from skorch.callbacks.lr_scheduler import LRScheduler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ModelCheckpoint:
    def __init__(self):
        self.model_path = '/tmp/' + '{:%H_%M_%S-%d-%m-%Y}'.format(
            datetime.now()) + '.model'
        self.best_metric_value = None

    def save_model(self, metric_value, model, low_better):
        if self.best_metric_value is None:
            torch.save(model, self.model_path)
            self.best_metric_value = metric_value

        if low_better:
            if metric_value < self.best_metric_value:
                torch.save(model, self.model_path)
                print(
                    'Model checkpoint: Model saved as {}. Current metric value '
                    '{:.4f} < {:.4f}'.format(
                        self.model_path, metric_value, self.best_metric_value))
                self.best_metric_value = metric_value
        else:
            if metric_value > self.best_metric_value:
                torch.save(model, self.model_path)
                print(
                    'Model checkpoint: Model saved as {}. Current metric value '
                    '{:.4f} > {:.4f}'.format(
                        self.model_path, metric_value, self.best_metric_value))
                self.best_metric_value = metric_value

    def load_best_model(self):
        print('Loading saved model.')
        return torch.load(self.model_path)


def train(net, dataset_train, dataset_val, dataset_test, lr=0.001,
          batch_size=200, epochs=100, device='cpu'):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                                 shuffle=False)

    accuracy_epochs_train = []
    accuracy_epochs_val = []
    loss_epochs_train = []
    loss_epochs_val = []

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    model_checkpointer = ModelCheckpoint()

    try:
        net.train()
        for epoch in range(epochs):
            # TRAIN
            loss = 0
            loss_one_epoch_train = []
            accuracy_one_epoch_train = []
            batches_len = []
            for batch in dataloader_train:
                x, y = batch
                batches_len.append(x.shape[0])
                y_pred = net(x.float().to(device))
                loss = loss_fn(y_pred, y.long().to(device))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                accuracy_one_epoch_train.append(accuracy_score(
                    y, np.argmax(y_pred.cpu().detach().numpy(), axis=1)))
                loss_one_epoch_train.append(loss.cpu().detach().numpy())

            scheduler.step()
            accuracy_epochs_train.append(np.average(accuracy_one_epoch_train,
                                                    weights=batches_len))
            loss_epochs_train.append(np.average(loss_one_epoch_train,
                                                weights=batches_len))

            # VALIDATION
            net.eval()
            loss_one_epoch_val = []
            accuracy_one_epoch_val = []
            batches_len = []
            with torch.no_grad():
                for batch in dataloader_val:
                    x, y = batch
                    batches_len.append(x.shape[0])
                    y_pred_val = net(x.float().to(device))
                    accuracy_one_epoch_val.append(accuracy_score(
                        y,
                        np.argmax(y_pred_val.cpu().detach().numpy(), axis=1)))
                    loss_one_epoch_val.append(loss_fn(
                        y_pred_val, y.long().to(device)).cpu().detach().numpy())

            accuracy_epochs_val.append(np.average(accuracy_one_epoch_val,
                                                  weights=batches_len))
            loss_epochs_val.append(np.average(loss_one_epoch_val,
                                              weights=batches_len))

            print('Epoch: ', epoch,
                  ' Train loss: {:.4f}'.format(loss_epochs_train[-1]),
                  ' Train accuracy {:.4f}'.format(accuracy_epochs_train[-1]),
                  ' Val loss: {:.4f}'.format(loss_epochs_val[-1]),
                  ' Val accuracy: {:.4f}'.format(accuracy_epochs_val[-1]))
            model_checkpointer.save_model(accuracy_epochs_val[-1], net,
                                          low_better=False)

    except KeyboardInterrupt:
        pass
    # TEST
    net = model_checkpointer.load_best_model()
    net.eval()
    accuracy_test, loss_test = [], []
    batches_len = []
    with torch.no_grad():
        for batch in dataloader_test:
            x, y = batch
            batches_len.append(x.shape[0])
            y_pred_val = net(x.float().to(device))
            accuracy_test.append(accuracy_score(
                y, np.argmax(y_pred_val.cpu().detach().numpy(), axis=1)))
            loss_test.append(loss_fn(
                y_pred_val, y.long().to(device)).cpu().detach().numpy())
        accuracy_test = np.average(accuracy_test, weights=batches_len)
        loss_test = np.average(loss_test, weights=batches_len)
    print('TEST loss: {:.2f}'.format(loss_test),
          ' TEST accuracy: {:.2f}'.format(accuracy_test))
    return net, dict(accuracy_train=accuracy_epochs_train,
                     loss_train=loss_epochs_train,
                     accuracy_val=accuracy_epochs_val,
                     loss_val=loss_epochs_val,
                     accuracy_test=accuracy_test,
                     loss_test=loss_test)


class EEGDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrainTestSplit(object):
    def __init__(self, train_size):
        assert isinstance(train_size, (int, float))
        self.train_size = train_size

    def __call__(self, dataset, y, **kwargs):
        if isinstance(self.train_size, int):
            n_train_samples = self.train_size
        else:
            n_train_samples = int(self.train_size * len(dataset))

        X, y = dataset.X, dataset.y
        return (
            EEGDataSet(X[:n_train_samples], y[:n_train_samples]),
            EEGDataSet(X[n_train_samples:], y[n_train_samples:]),
        )


data_folder = "/home/maciej/data/bci_competition"
subject_id = 1  # 1-9
low_cut_hz = 4  # 0 or 4
cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
ival = [-500, 4000]
input_time_length = 1000
max_epochs = 5
max_increase_epochs = 80
batch_size = 60
high_cut_hz = 38
factor_new = 1e-3
init_block_size = 1000
valid_set_fraction = 0.2

train_filename = "A{:02d}T.gdf".format(subject_id)
test_filename = "A{:02d}E.gdf".format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace(".gdf", ".mat")
test_label_filepath = test_filepath.replace(".gdf", ".mat")

train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath
)
test_loader = BCICompetition4Set2A(test_filepath,
                                   labels_filename=test_label_filepath)
raw_train = train_loader.load()
raw_test = test_loader.load()

# Preprocessing

raw_train = raw_train.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_train.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
raw_train = mne_apply(lambda a: a * 1e6, raw_train)
raw_train = mne_apply(
    lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, raw_train.info["sfreq"], filt_order=3,
        axis=1
    ),
    raw_train,
)
raw_train = mne_apply(
    lambda a: exponential_running_standardize(
        a.T, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ).T,
    raw_train,
)

raw_test = raw_test.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(raw_test.ch_names) == 22
raw_test = mne_apply(lambda a: a * 1e6, raw_test)
raw_test = mne_apply(
    lambda a: bandpass_cnt(
        a, low_cut_hz, high_cut_hz, raw_test.info["sfreq"], filt_order=3, axis=1
    ),
    raw_test,
)
raw_test = mne_apply(
    lambda a: exponential_running_standardize(
        a.T, factor_new=factor_new, init_block_size=init_block_size, eps=1e-4
    ).T,
    raw_test,
)
marker_def = OrderedDict(
    [("Left Hand", [1]), ("Right Hand", [2],), ("Foot", [3]), ("Tongue", [4])]
)

train_set = create_signal_target_from_raw_mne(raw_train, marker_def, ival)
test_set = create_signal_target_from_raw_mne(raw_test, marker_def, ival)

train_valid_set = EEGDataSet(train_set.X, train_set.y)
test_set = EEGDataSet(test_set.X, test_set.y)

splitter = TrainTestSplit(train_size=0.8)

ds_train, ds_valid = splitter(train_valid_set, None)

model_old = Deep4Net(22, n_classes=4, input_time_length=1125,
                     final_conv_length='auto')
if cuda:
    model_old.cuda()
    device = 'cuda'

_ = train(model_old, dataset_train=ds_train, dataset_val=ds_valid,
          dataset_test=test_set, lr=0.001, batch_size=batch_size, epochs=50,
          device=device)

cp = Checkpoint('valid_acc_best')

model = Deep4Net(22, n_classes=4, input_time_length=1125,
                 final_conv_length='auto')
if cuda:
    model.cuda()
    device = 'cuda'

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=AdamW,
    optimizer__lr=0.001,
    optimizer__weight_decay=0.0001,
    train_split=splitter,
    batch_size=batch_size,
    lr=0.001,
    callbacks=[
        ('train_acc', EpochScoring(
            'accuracy', name='train_accuracy', on_train=True,
            lower_is_better=False)),
        ('valid_acc', EpochScoring(
            'accuracy', name='valid_acc', on_train=False,
            lower_is_better=False)),
        ('Checkpoint', cp),
        ('lr', LRScheduler(policy='CosineAnnealingLR', T_max=50)),
        ('progress', ProgressBar())
    ],
    iterator_train=DataLoader,
    iterator_train__shuffle=True,
    iterator_valid=DataLoader,
    iterator_valid__shuffle=False,
    device=device,
    warm_start=True
)

clf.fit(train_valid_set, y=None, epochs=50)
clf.load_params(checkpoint=cp)

accuracy_score(test_set.y, clf.predict(test_set))
