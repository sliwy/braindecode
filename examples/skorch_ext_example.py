import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier, NeuralNet
from skorch.callbacks import EpochScoring, LRScheduler, EpochTimer, \
    BatchScoring, PrintLog
from skorch.utils import train_loss_score, valid_loss_score, noop
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader

from braindecode.models import Deep4Net

X_train_valid = np.load('train_set_X.npy')
y_train_valid = np.load('train_set_y.npy')


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        if self.X.ndim == 3:
            self.X = self.X[:, :, :, None]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrainTestSplit:
    def __init__(self, train_size):
        assert isinstance(train_size, (int, float))
        self.train_size = train_size

    def __call__(self, dataset, y, **kwargs):
        idxs_train, idxs_test = train_test_split(np.arange(len(dataset)),
                                                 train_size=self.train_size,
                                                 shuffle=False)

        return Subset(dataset, idxs_train), Subset(dataset, idxs_test)
    
    
def train(net, dataset_train, dataset_val, lr=0.001, batch_size=200, epochs=100,
          device='cpu'):
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=False)

    accuracy_epochs_train = []
    accuracy_epochs_val = []
    loss_epochs_train = []
    loss_epochs_val = []

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    try:
        net.train()
        for epoch in range(epochs):
            # TRAIN
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
                    y, np.argmax(y_pred.cpu().detach().numpy(), axis=1))
                )
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

    except KeyboardInterrupt:
        pass
    # TEST
    net.eval()
    return net, dict(accuracy_train=accuracy_epochs_train,
                     loss_train=loss_epochs_train,
                     accuracy_val=accuracy_epochs_val,
                     loss_val=loss_epochs_val)


train_valid_set = MyDataset(X_train_valid, y_train_valid)

splitter = TrainTestSplit(train_size=0.8)

ds_train, ds_valid = splitter(train_valid_set, None)

model_old = Deep4Net(22, n_classes=4, input_time_length=1125,
                     final_conv_length='auto')
model_skorch = Deep4Net(22, n_classes=4, input_time_length=1125,
                        final_conv_length='auto')

lr = 0.001
batch_size = 200
epochs = 50
device = 'cpu'
train_results = train(model_old, ds_train, ds_valid, lr=lr,
                      batch_size=batch_size, epochs=epochs, device=device)


class MyClassifier(NeuralNetClassifier):
    """Classifier that does not assume softmax activation.
    Calls loss function directly without applying log or anything.
    """

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch by calling NeuralNet get_loss.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.

        """
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)

    # Removes default EpochScoring callback computing 'accuracy' to work properly
    # with cropped decoding.
    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimer()),
            (
                "train_loss",
                BatchScoring(
                    train_loss_score,
                    name="train_loss",
                    on_train=True,
                    target_extractor=noop,
                ),
            ),
            (
                "valid_loss",
                BatchScoring(
                    valid_loss_score, name="valid_loss", target_extractor=noop,
                ),
            ),
            ("print_log", PrintLog()),
        ]


clf = MyClassifier(
    model_skorch,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.001,
    train_split=splitter,
    batch_size=batch_size,
    lr=lr,
    callbacks=[
        ('train_acc', EpochScoring(
            'accuracy', name='train_accuracy', on_train=True,
            lower_is_better=False)),
        ('valid_acc', EpochScoring(
            'accuracy', name='valid_acc', on_train=False,
            lower_is_better=False)),
        ('lr', LRScheduler(policy='CosineAnnealingLR', T_max=epochs))
    ],
    iterator_train=DataLoader,
    iterator_train__shuffle=True,
    iterator_valid=DataLoader,
    iterator_valid__shuffle=False,
    device=device,
    warm_start=False
)

clf.fit(train_valid_set, y=None, epochs=epochs)
