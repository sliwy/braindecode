from collections import OrderedDict
from datetime import datetime

import numpy as np
from braindecode.datasets import MOABBDataset
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from skorch.callbacks import Checkpoint
from tqdm.auto import tqdm


def main(model_name, output_path, params):
    results_subjects = OrderedDict()
    results_subjects['params'] = params
    t_str = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
    for subject_id in tqdm(range(1, 10)):

        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

        from braindecode.preprocessing import (
            exponential_moving_standardize, preprocess, Preprocessor, scale)

        low_cut_hz = 4.  # low cut frequency for filtering
        high_cut_hz = 38.  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000

        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                         factor_new=factor_new, init_block_size=init_block_size)
        ]

        # Transform the data
        preprocess(dataset, preprocessors)

        from braindecode.preprocessing import create_windows_from_events

        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )

        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        test_set = splitted['session_E']

        import torch
        from braindecode.util import set_random_seeds
        from braindecode.models import ShallowFBCSPNet, Deep4Net

        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True
        # Set random seed to be able to roughly reproduce results
        # Note that with cudnn benchmark set to True, GPU indeterminism
        # may still make results substantially different between runs.
        # To obtain more consistent results at the cost of increased computation time,
        # you can set `cudnn_benchmark=False` in `set_random_seeds`
        # or remove `torch.backends.cudnn.benchmark = True`
        seed = 20200220
        set_random_seeds(seed=seed, cuda=cuda)

        n_classes = 4
        # Extract number of chans and time steps from dataset
        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]

        if model_name == 'ShallowFBCSPNet':
            model = ShallowFBCSPNet(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
            lr = 0.0625 * 0.01
            weight_decay = 0
        elif model_name == 'Deep4Net':
            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
            lr = 1 * 0.01
            weight_decay = 0.5 * 0.001

        elif model_name == 'EEGNetv1':
            raise ValueError

        # Send model to GPU
        if cuda:
            model.cuda()

        from skorch.callbacks import LRScheduler
        from skorch.helper import predefined_split

        from braindecode import EEGClassifier

        batch_size = 64
        n_epochs = 100
        cp = Checkpoint(dirname=model_name + t_str)
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(test_set),  # using valid_set for validation
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                # ('early_stopping', EarlyStopping(patience=20)),
                # ('checkpoint', cp)

            ],
            device=device,
        )
        # Model training for a specified number of epochs. `y` is None as it is already supplied
        # in the dataset.
        clf.fit(train_set, y=None, epochs=n_epochs)
        # clf.load_params(checkpoint=cp)

        # add class labels
        # label_dict is class_name : str -> i_class : int
        label_dict = list(test_set.datasets[0].windows.event_id.items())

        results = OrderedDict()
        results['labels'] = label_dict

        results['accuracy'] = {}
        results['balanced_accuracy'] = {}
        results['confusion_matrix'] = {}
        results['predictions'] = {}
        for ds_name, ds in [('train', train_set),
                            ('test', test_set)]:
            array = np.stack([ds[i][0] for i in range(len(ds))])
            y_array = np.array([ds[i][1] for i in range(len(ds))])
            preds = clf.predict(ds)
            results['accuracy'][ds_name] = accuracy_score(y_array, preds)
            results['balanced_accuracy'][ds_name] = balanced_accuracy_score(y_array, preds)
            results['confusion_matrix'][ds_name] = confusion_matrix(y_array, preds)
            results['predictions'][ds_name] = preds
        results_subjects[subject_id] = results
        torch.save(results_subjects,
                   output_path + 'results' + model_name.replace('+', '_') + '_' + t_str + '.pkl',
                   pickle_protocol=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        choices=['ShallowFBCSPNet', 'Deep4Net'], required=True)
    parser.add_argument('--output_path', default='./')

    args = parser.parse_args()
    params = vars(args)

    main(args.model_name, args.output_path, params)
