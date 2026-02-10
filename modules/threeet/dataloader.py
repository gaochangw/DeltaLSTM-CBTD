"""
DataLoader for 3ET+ Event-Based Eye Tracking dataset.
Uses numpy memory-mapped files for near-zero RAM usage.
Only loads batch-sized chunks from disk on demand via OS page cache.
"""
import os
import json
import math
import numpy as np
import torch
import utils.util as util


class CustomDataLoader:
    def __init__(self, trainfile, valfile, testfile, args, **kwargs):
        self.traindir = trainfile   # now a directory path, not .h5
        self.valdir = valfile
        self.testdir = testfile
        self.qa = args.qa
        self.aqi = args.aqi
        self.aqf = args.aqf
        self.normalization = kwargs.get('normalization', 0)
        # JaneEye-Net augmentation and normalization options
        self.voxel_ch_norm = getattr(args, 'voxel_ch_norm', 0)
        self.augment_hflip = getattr(args, 'augment_hflip', 0.0)
        self.augment_noise = getattr(args, 'augment_noise', 0.0)
        self.augment_noise_min = getattr(args, 'augment_noise_min', 0.05)
        self.augment_noise_max = getattr(args, 'augment_noise_max', 2.2)

        # Read metadata from each split directory
        train_meta = self._load_metadata(self.traindir)
        val_meta = self._load_metadata(self.valdir)
        test_meta = self._load_metadata(self.testdir)

        self.n_features = train_meta['n_features']
        self.n_classes = train_meta['n_classes']
        self.num_sample_train = train_meta['n_samples']
        self.num_sample_val = val_meta['n_samples']
        self.num_sample_test = test_meta['n_samples']

        # Memory-map the .npy files (zero RAM cost, OS pages in on demand)
        self._train_features = np.load(os.path.join(self.traindir, 'features.npy'), mmap_mode='r')
        self._train_targets = np.load(os.path.join(self.traindir, 'targets.npy'), mmap_mode='r')
        self._val_features = np.load(os.path.join(self.valdir, 'features.npy'), mmap_mode='r')
        self._val_targets = np.load(os.path.join(self.valdir, 'targets.npy'), mmap_mode='r')
        self._test_features = np.load(os.path.join(self.testdir, 'features.npy'), mmap_mode='r')
        self._test_targets = np.load(os.path.join(self.testdir, 'targets.npy'), mmap_mode='r')

        # Slice to actual sample count (memmap may be larger if errors during extraction)
        self._train_features = self._train_features[:self.num_sample_train]
        self._train_targets = self._train_targets[:self.num_sample_train]
        self._val_features = self._val_features[:self.num_sample_val]
        self._val_targets = self._val_targets[:self.num_sample_val]
        self._test_features = self._test_features[:self.num_sample_test]
        self._test_targets = self._test_targets[:self.num_sample_test]

        # Compute normalization stats if needed (chunked, no full load)
        if self.normalization:
            self._compute_norm_stats()

        print(f"Loaded (memmap): Train={self.num_sample_train}, Val={self.num_sample_val}, Test={self.num_sample_test}")
        print(f"Feature shape: {self._train_features.shape[1:]} (T, C, H, W)")
        print(f"n_features={self.n_features}, n_classes={self.n_classes}")

    def _load_metadata(self, dirpath):
        """Load metadata.json from a split directory."""
        meta_path = os.path.join(dirpath, 'metadata.json')
        with open(meta_path, 'r') as f:
            return json.load(f)

    def _compute_norm_stats(self):
        """Compute per-channel mean/std from training data using chunked Welford algorithm.
        Reads memmap in chunks to avoid loading entire dataset into RAM.
        """
        chunk_size = 256
        n_chunks = math.ceil(self.num_sample_train / chunk_size)

        n_total = 0
        running_sum = None
        running_sq_sum = None

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.num_sample_train)
            chunk = np.array(self._train_features[start:end])  # materialize chunk

            # Sum over all axes except channel: axes (0,1,3,4)
            chunk_sum = chunk.sum(axis=(0, 1, 3, 4))
            chunk_sq_sum = (chunk ** 2).sum(axis=(0, 1, 3, 4))
            chunk_count = chunk.shape[0] * chunk.shape[1] * chunk.shape[3] * chunk.shape[4]

            if running_sum is None:
                running_sum = chunk_sum.astype(np.float64)
                running_sq_sum = chunk_sq_sum.astype(np.float64)
            else:
                running_sum += chunk_sum.astype(np.float64)
                running_sq_sum += chunk_sq_sum.astype(np.float64)
            n_total += chunk_count

        mean = running_sum / n_total
        var = (running_sq_sum / n_total) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))
        std = np.where(std == 0, 1, std)

        # Reshape to (1, 1, C, 1, 1) for broadcasting
        self.mean_train = mean.reshape(1, 1, -1, 1, 1).astype(np.float32)
        self.std_train = std.reshape(1, 1, -1, 1, 1).astype(np.float32)

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

        # Select memmap arrays and sample count
        if set_name == 'train':
            features = self._train_features
            targets = self._train_targets
            n_samples = self.num_sample_train
        elif set_name == 'val':
            features = self._val_features
            targets = self._val_targets
            n_samples = self.num_sample_val
        elif set_name == 'test':
            features = self._test_features
            targets = self._test_targets
            n_samples = self.num_sample_test
        else:
            raise RuntimeError('Please select a valid set.')

        # Shuffle indices (tiny array, ~200 KB for 50k samples)
        if shuffle_type == 'random':
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        # Create batches — read from memmap on-the-fly
        n_batches = int(math.ceil(n_samples / batch_size))

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Sort indices for sequential disk access (better page cache behavior)
            sort_order = np.argsort(batch_indices)
            sorted_indices = batch_indices[sort_order]

            # Fancy indexing on memmap materializes only this batch (~35 MB)
            batch_features = np.array(features[sorted_indices])  # (B, T, C, H, W)
            batch_targets = np.array(targets[sorted_indices])    # (B, T, 3)

            # Restore original shuffled order
            unsort_order = np.argsort(sort_order)
            batch_features = batch_features[unsort_order]
            batch_targets = batch_targets[unsort_order]

            # Ensure float32
            batch_features = batch_features.astype(np.float32)
            batch_targets = batch_targets.astype(np.float32)

            # Normalize features
            if self.normalization:
                batch_features = (batch_features - self.mean_train) / self.std_train

            # Add Gaussian noise for training
            if enable_gauss > 0 and set_name == 'train':
                noise = enable_gauss * np.random.standard_normal(batch_features.shape)
                batch_features = batch_features + noise.astype(np.float32)

            # Per-channel non-zero voxel normalization (JaneEye-Net style)
            if self.voxel_ch_norm:
                # batch_features shape: (B, T, C, H, W)
                n_channels = batch_features.shape[2]
                for c in range(n_channels):
                    ch_data = batch_features[:, :, c, :, :]
                    nonzero_mask = ch_data != 0
                    if nonzero_mask.any():
                        nz_vals = ch_data[nonzero_mask]
                        ch_mean = nz_vals.mean()
                        ch_std = nz_vals.std()
                        if ch_std > 0:
                            batch_features[:, :, c, :, :] = np.where(
                                nonzero_mask,
                                (ch_data - ch_mean) / ch_std,
                                ch_data)

            # Data augmentation (training only)
            if set_name == 'train':
                # Horizontal flip
                if self.augment_hflip > 0 and np.random.random() < self.augment_hflip:
                    batch_features = batch_features[:, :, :, :, ::-1].copy()
                    batch_targets[:, :, 0] = 1.0 - batch_targets[:, :, 0]  # flip x coord

                # Gaussian noise with random scaling
                if self.augment_noise > 0 and np.random.random() < self.augment_noise:
                    factor = np.random.uniform(self.augment_noise_min, self.augment_noise_max)
                    noise = np.random.standard_normal(batch_features.shape).astype(np.float32)
                    batch_features = batch_features + factor * noise

            # Convert to tensors
            # Shape for RNN: (T, B, C, H, W) - transpose from (B, T, C, H, W)
            batch_features = torch.from_numpy(batch_features).float().transpose(0, 1)
            batch_targets_full = torch.from_numpy(batch_targets).float()  # (B, T, 3)

            # Quantize features
            batch_features = util.quantize_tensor(batch_features, self.aqi, self.aqf, self.qa)

            # Feature lengths (all same for fixed-length sequences)
            batch_size_actual = batch_features.shape[1]
            seq_len = batch_features.shape[0]
            feature_lengths = torch.full((batch_size_actual,), seq_len, dtype=torch.int32)

            dict_batch_data = {
                'features': batch_features,           # (T, B, C, H, W)
                'feature_lengths': feature_lengths,
                'targets': batch_targets_full[:, :, :2],   # (B, T, 2) - x, y only
                'targets_full': batch_targets_full,        # (B, T, 3) - x, y, close
                'targets_metric': batch_targets_full,      # (B, T, 3) for metric computation
                'labels': batch_targets_full[:, :, :2],    # Alias for compatibility
                'target_lengths': torch.full((batch_size_actual,), seq_len, dtype=torch.int32)
            }

            yield dict_batch_data
