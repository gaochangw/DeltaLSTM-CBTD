import math
import numpy as np
import h5py
import utils.util as util
import torch
from torch.nn.utils.rnn import pad_sequence


def add_gaussian_noise(batch_feats, sigma=0.6):
    batch_gauss = []
    for sample in batch_feats:
        noise_mat = sigma * np.random.standard_normal(sample.shape)
        sample = sample + noise_mat

        batch_gauss.append(sample)
    return batch_gauss


def create_batch_idx(s_idx, feature_lens, cache_size, batch_size, mode='batch'):
    list_batches = []
    batch = []
    max_len = 0

    if mode == 'batch':
        for i, sample in enumerate(s_idx):
            if len(batch) < batch_size:
                batch.append(sample)
            else:
                list_batches.append(batch)
                batch = [sample]
    elif mode == 'cache':
        for i, sample in enumerate(s_idx):
            max_len = max(feature_lens[sample], max_len)
            num_frames = (len(batch) + 1) * max_len
            if num_frames <= cache_size:
                batch.append(sample)
            else:
                list_batches.append(batch)
                batch = [sample]
                max_len = feature_lens[sample]

    list_batches.append(batch)
    return list_batches


def idx_to_slice(lens):
    idx = []
    lens_cs = torch.cumsum(lens, dim=0)
    for i, len in enumerate(lens):
        idx.append((lens_cs[i] - lens[i], lens_cs[i]))
    return idx


class CustomDataLoader(object):
    def __init__(self, trainfile, valfile, testfile, args, **kwargs):
        self.mean_train = 0
        self.std_train = 0
        self.trainfile = trainfile
        self.valfile = valfile
        self.testfile = testfile
        self.qa = args.qa
        self.aqi = args.aqi
        self.aqf = args.aqf
        self.normalization = kwargs['normalization']

        # Evaluate train set
        with h5py.File(self.trainfile, 'r') as hf:
            self.train_features = np.array(hf.get('features')).astype(np.float32)
            self.train_feature_lengths = np.array(hf.get('feature_lengths')).astype(int)
            self.train_targets_61 = np.array(hf.get('targets_61')).astype(int)
            self.train_targets_48 = np.array(hf.get('targets_48')).astype(int)
            self.train_targets_39 = np.array(hf.get('targets_39')).astype(int)
            self.train_target_lengths = np.array(hf.get('target_lengths')).astype(int)
            self.n_features = int(np.asarray(hf.get('n_features')).astype(int))
            self.n_classes = int(np.asarray(hf.get('n_classes')).astype(int))

            # Stat for each feature
            self.mean_train = np.mean(self.train_features[:, :], axis=0, keepdims=True)
            self.std_train = np.std(self.train_features[:, :], axis=0, keepdims=True)
            # self.mean_train = np.mean(self.train_features[:, :])
            # self.std_train = np.std(self.train_features[:, :])
            # self.max_train = np.amax(self.train_features[:, :], axis=0)
            # self.min_train = np.amin(self.train_features[:, :], axis=0)
            # self.mean_train = np.mean(self.train_features[:, :], axis=0)
            # self.std_train = np.std(self.train_features[:, :], axis=0)
            self.max_train = np.amax(self.train_features[:, :])
            self.min_train = np.amin(self.train_features[:, :])
            feature_lengths = np.asarray(hf.get('feature_lengths')).astype(int)
            self.num_sample_train = feature_lengths.size

        # Evaluate val set
        with h5py.File(self.valfile, 'r') as hf:
            self.val_features = np.array(hf.get('features')).astype(np.float32)
            self.val_feature_lengths = np.array(hf.get('feature_lengths')).astype(int)
            self.val_targets_61 = np.array(hf.get('targets_61')).astype(int)
            self.val_targets_48 = np.array(hf.get('targets_48')).astype(int)
            self.val_targets_39 = np.array(hf.get('targets_39')).astype(int)
            self.val_target_lengths = np.array(hf.get('target_lengths')).astype(int)
            # Stat for each feature
            self.mean_val = np.mean(self.val_features[:, :], axis=0)
            self.std_val = np.std(self.val_features[:, :], axis=0)
            self.max_val = np.amax(self.val_features[:, :], axis=0)
            self.min_val = np.amin(self.val_features[:, :], axis=0)
            feature_lengths = np.asarray(hf.get('feature_lengths')).astype(int)
            self.num_sample_val = feature_lengths.size

        # Evaluate test set
        with h5py.File(self.testfile, 'r') as hf:
            self.test_features = np.array(hf.get('features')).astype(np.float32)
            self.test_feature_lengths = np.array(hf.get('feature_lengths')).astype(int)
            self.test_targets_61 = np.array(hf.get('targets_61')).astype(int)
            self.test_targets_48 = np.array(hf.get('targets_48')).astype(int)
            self.test_targets_39 = np.array(hf.get('targets_39')).astype(int)
            self.test_target_lengths = np.array(hf.get('target_lengths')).astype(int)
            # Stat for each feature
            self.mean_test = np.mean(self.test_features[:, :], axis=0)
            self.std_test = np.std(self.test_features[:, :], axis=0)
            self.max_test = np.amax(self.test_features[:, :], axis=0)
            self.min_test = np.amin(self.test_features[:, :], axis=0)
            feature_lengths = np.asarray(hf.get('feature_lengths')).astype(int)
            self.num_sample_test = feature_lengths.size

    def get_num_batch(self, set_name, batch_size):
        if set_name == 'train':
            return int(math.ceil(float(self.num_sample_train) / float(batch_size)))
        elif set_name == 'val':
            return int(math.ceil(float(self.num_sample_val) / float(batch_size)))
        elif set_name == 'test':
            return int(math.ceil(float(self.num_sample_test) / float(batch_size)))

    def get_num_sample(self, set_name):
        if set_name == 'train':
            return self.num_sample_train
        elif set_name == 'val':
            return self.num_sample_val
        elif set_name == 'test':
            return self.num_sample_test

    def iterate(self,
                epoch,
                set_name,
                cache_size=1000,
                batch_size=32,
                mode='batch',
                shuffle_type='high_throughput',
                enable_gauss=0,
                ctc_mode='first',
                **kwargs):

        np.random.seed(epoch)

        if ctc_mode == 'first':
            ctc_label_shift = 1
        else:
            ctc_label_shift = 0

        # Select datasets
        if set_name == 'train':
            features = torch.from_numpy(self.train_features).float()
            feature_lengths = torch.from_numpy(self.train_feature_lengths).int()
            targets_61 = torch.from_numpy(self.train_targets_61).int()
            targets_48 = torch.from_numpy(self.train_targets_48).int()
            targets_39 = torch.from_numpy(self.train_targets_39).int()
            target_lengths = torch.from_numpy(self.train_target_lengths).int()
        elif set_name == 'val':
            features = torch.from_numpy(self.val_features).float()
            feature_lengths = torch.from_numpy(self.val_feature_lengths).int()
            targets_61 = torch.from_numpy(self.val_targets_61).int()
            targets_48 = torch.from_numpy(self.val_targets_48).int()
            targets_39 = torch.from_numpy(self.val_targets_39).int()
            target_lengths = torch.from_numpy(self.val_target_lengths).int()
        elif set_name == 'test':
            features = torch.from_numpy(self.test_features).float()
            feature_lengths = torch.from_numpy(self.test_feature_lengths).int()
            targets_61 = torch.from_numpy(self.test_targets_61).int()
            targets_48 = torch.from_numpy(self.test_targets_48).int()
            targets_39 = torch.from_numpy(self.test_targets_39).int()
            target_lengths = torch.from_numpy(self.test_target_lengths).int()
        else:
            raise RuntimeError('Please select a valid set.')

        # Get index slices for features and targets
        feat_slice_idx = idx_to_slice(feature_lengths)
        label_slice_idx = idx_to_slice(target_lengths)

        # Constants
        n_samples = len(feature_lengths)
        ndim = len(features[1, :])

        # Batch Shuffle
        if shuffle_type == 'high_throughput':
            s_idx = np.argsort(feature_lengths.numpy())
        elif shuffle_type == 'random':
            s_idx = np.random.permutation(n_samples)
        else:
            s_idx = range(n_samples)
        batches_idx = create_batch_idx(s_idx=s_idx,
                                            feature_lens=feature_lengths,
                                            cache_size=cache_size,
                                            batch_size=batch_size,
                                            mode=mode)

        n_batches = len(batches_idx)  # Number of batches
        b = 0
        while b < n_batches:
            curr_batch_idx = batches_idx[b]

            # Load batch
            batch_feats = []
            batch_labels_61 = []
            batch_labels_48 = []
            batch_labels_39 = []
            for sample_idx in curr_batch_idx:
                batch_feats.append(features[self.slc(sample_idx, feat_slice_idx), :])
                batch_labels_61.append(targets_61[self.slc(sample_idx, label_slice_idx)])
                batch_labels_48.append(targets_48[self.slc(sample_idx, label_slice_idx)])
                batch_labels_39.append(targets_39[self.slc(sample_idx, label_slice_idx)])

            # Normalize batch
            if self.normalization != 0:
                batch_feats = self.normalize(batch_feats, self.mean_train, self.std_train)

            # Add gaussian noise:
            if enable_gauss != 0.0:
                batch_feats = add_gaussian_noise(batch_feats, sigma=enable_gauss)

            # Zero Padding
            b_lenX = feature_lengths[curr_batch_idx]
            bY_61 = torch.cat(batch_labels_61) + int(ctc_label_shift)
            bY_48 = torch.cat(batch_labels_48) + int(ctc_label_shift)
            bY_39 = torch.cat(batch_labels_39) + int(ctc_label_shift)
            b_lenY = torch.tensor([len(label_48) for label_48 in batch_labels_48], dtype=torch.int32)
            bX = pad_sequence(batch_feats)
            b += 1

            # Quantize Features
            bX = util.quantize_tensor(bX, self.aqi, self.aqf, self.qa)

            dict_batch_data = {'features': bX,
                               'feature_lengths': b_lenX,
                               'targets_39': bY_39,
                               'targets_48': bY_48,
                               'targets_61': bY_61,
                               'target_lengths': b_lenY}

            yield dict_batch_data

    def slc(self, i, idx):
        return slice(idx[i][0], idx[i][1])

    def normalize(self, batch, ep_mean, ep_std):
        batch_normalized = []
        for sample in batch:
            sample = sample - ep_mean
            sample = sample / ep_std
            batch_normalized.append(sample)
        return batch_normalized
