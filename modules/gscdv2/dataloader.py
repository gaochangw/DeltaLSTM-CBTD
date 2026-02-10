"""
DataLoader for Google Speech Commands Dataset v2 - Keyword Spotting.
Handles batching and data loading for classification task.
"""
import math
import numpy as np
import h5py
import torch
import utils.util as util


class CustomDataLoader:
    def __init__(self, trainfile, valfile, testfile, args, **kwargs):
        self.trainfile = trainfile
        self.valfile = valfile
        self.testfile = testfile
        self.qa = args.qa
        self.aqi = args.aqi
        self.aqf = args.aqf
        self.normalization = kwargs.get('normalization', 1)
        self.n_classes = args.n_classes

        # Load train set
        with h5py.File(self.trainfile, 'r') as hf:
            self.train_features = np.array(hf.get('features')).astype(np.float32)  # (N, T, F)
            self.train_labels = np.array(hf.get('labels')).astype(np.int64)
            self.n_features = int(np.asarray(hf.get('n_features')))
            self.n_classes = int(np.asarray(hf.get('n_classes')))

            # Compute statistics for normalization
            self.mean_train = np.mean(self.train_features, axis=(0, 1), keepdims=True)
            self.std_train = np.std(self.train_features, axis=(0, 1), keepdims=True)
            self.std_train = np.where(self.std_train == 0, 1, self.std_train)

            self.num_sample_train = len(self.train_labels)

        # Load validation set
        with h5py.File(self.valfile, 'r') as hf:
            self.val_features = np.array(hf.get('features')).astype(np.float32)
            self.val_labels = np.array(hf.get('labels')).astype(np.int64)
            self.num_sample_val = len(self.val_labels)

        # Load test set
        with h5py.File(self.testfile, 'r') as hf:
            self.test_features = np.array(hf.get('features')).astype(np.float32)
            self.test_labels = np.array(hf.get('labels')).astype(np.int64)
            self.num_sample_test = len(self.test_labels)

        print(f"Loaded: Train={self.num_sample_train}, Val={self.num_sample_val}, Test={self.num_sample_test}")
        print(f"Feature shape: {self.train_features.shape[1:]} (T, F)")
        print(f"n_features={self.n_features}, n_classes={self.n_classes}")

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
                shuffle_type='random',
                enable_gauss=0,
                ctc_mode='first',
                **kwargs):

        np.random.seed(epoch)

        # Select dataset
        if set_name == 'train':
            features = self.train_features.copy()
            labels = self.train_labels.copy()
            n_samples = self.num_sample_train
        elif set_name == 'val':
            features = self.val_features.copy()
            labels = self.val_labels.copy()
            n_samples = self.num_sample_val
        elif set_name == 'test':
            features = self.test_features.copy()
            labels = self.test_labels.copy()
            n_samples = self.num_sample_test
        else:
            raise RuntimeError('Please select a valid set.')

        # Normalize features
        if self.normalization:
            features = (features - self.mean_train) / self.std_train

        # Shuffle indices
        if shuffle_type == 'random':
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        # Create batches
        n_batches = int(math.ceil(n_samples / batch_size))

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_features = features[batch_indices]  # (B, T, F)
            batch_labels = labels[batch_indices]  # (B,)

            # Add Gaussian noise for training
            if enable_gauss > 0 and set_name == 'train':
                noise = enable_gauss * np.random.standard_normal(batch_features.shape)
                batch_features = batch_features + noise

            # Convert to tensors
            # Shape for RNN: (T, B, F) - transpose from (B, T, F)
            batch_features = torch.from_numpy(batch_features).float().transpose(0, 1)
            batch_labels = torch.from_numpy(batch_labels).long()

            # Quantize features
            batch_features = util.quantize_tensor(batch_features, self.aqi, self.aqf, self.qa)

            # Feature lengths (all same for fixed-length audio)
            feature_lengths = torch.full((len(batch_labels),), batch_features.shape[0], dtype=torch.int32)

            dict_batch_data = {
                'features': batch_features,  # (T, B, F)
                'feature_lengths': feature_lengths,
                'labels': batch_labels,  # (B,)
                # For compatibility with TIMIT code structure
                'targets': batch_labels,
                'targets_39': batch_labels,
                'targets_48': batch_labels,
                'targets_61': batch_labels,
                'target_lengths': torch.ones(len(batch_labels), dtype=torch.int32)
            }

            yield dict_batch_data
