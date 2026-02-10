"""
Feature extractor for 3ET+ Event-Based Eye Tracking dataset.
Converts raw events to voxel grids and saves as numpy memmap files.
Two-pass approach: count sequences first, then fill pre-allocated memmaps.
Peak RAM: ~200 MB (one recording at a time).
"""
import os
import json
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from modules.threeet import log


def custom_to_voxel_grid_numpy(events_x, events_y, events_t, events_p,
                                sensor_width, sensor_height, n_time_bins=3):
    """Build a voxel grid with bilinear interpolation in the time domain.

    Ported from 3ET+ custom_transforms.py (Zhu et al. 2019).

    Args:
        events_x: x coordinates (int array)
        events_y: y coordinates (int array)
        events_t: timestamps (float array)
        events_p: polarity (-1/+1 float array)
        sensor_width: sensor width after spatial downsampling
        sensor_height: sensor height after spatial downsampling
        n_time_bins: number of temporal bins

    Returns:
        numpy array of shape (n_time_bins, sensor_height, sensor_width)
    """
    if len(events_t) == 0:
        return np.zeros((n_time_bins, sensor_height, sensor_width), dtype=np.float32)

    voxel_grid = np.zeros((n_time_bins, sensor_height, sensor_width), dtype=np.float64).ravel()

    # Normalize timestamps to [0, n_time_bins]
    time_diff = events_t[-1] - events_t[0]
    if time_diff < 1e-6:
        ts = np.zeros_like(events_t, dtype=np.float64)
    else:
        ts = n_time_bins * (events_t.astype(np.float64) - events_t[0]) / time_diff

    xs = events_x.astype(int)
    ys = events_y.astype(int)
    pols = events_p.astype(np.float64)

    # Bilinear interpolation in time
    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_width
        + tis[valid_indices] * sensor_width * sensor_height,
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_width
        + (tis[valid_indices] + 1) * sensor_width * sensor_height,
        vals_right[valid_indices],
    )

    voxel_grid = voxel_grid.reshape((n_time_bins, sensor_height, sensor_width))
    return voxel_grid.astype(np.float32)


class FeatExtractor:
    def __init__(self):
        pass

    def extract(self, args):
        """Extract voxel grid features from 3ET+ event data."""
        data_dir = os.path.join('data', args.dataset_name)

        # Load description CSV
        desc_path = os.path.join(data_dir, 'description.csv')
        if not os.path.exists(desc_path):
            raise RuntimeError(
                f"Description CSV not found at {desc_path}. Run data_prep first."
            )
        df = pd.read_csv(desc_path)

        # Parameters
        spatial_factor = getattr(args, 'spatial_factor', 0.125)
        n_time_bins = getattr(args, 'n_time_bins', 3)
        seq_length = getattr(args, 'seq_length', 30)
        train_stride = getattr(args, 'train_stride', 15)
        val_stride = getattr(args, 'val_stride', 30)
        sensor_width = getattr(args, 'sensor_width', 640)
        sensor_height = getattr(args, 'sensor_height', 480)
        temp_subsample_factor = getattr(args, 'temp_subsample_factor', 1.0)

        # Downsampled sensor dimensions
        ds_width = int(sensor_width * spatial_factor)
        ds_height = int(sensor_height * spatial_factor)

        # Get output directory names (no .h5 extension)
        _, traindir, valdir, testdir = log.gen_dataset_name(args)

        for split, output_dir_name in [('train', traindir), ('val', valdir), ('test', testdir)]:
            print(f"\nExtracting features for {split} set...")

            split_df = df[df['group'] == split]
            stride = train_stride if split == 'train' else val_stride

            output_dir = os.path.join('feat', args.dataset_name, output_dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # === Pass 1: Count total sequences ===
            print(f"  Pass 1: Counting sequences...")
            total_sequences = 0
            valid_rows = []
            for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                               desc=f'Count {split}'):
                label_path = row['label_path']
                h5_path = row['h5_path']

                if not os.path.exists(h5_path):
                    continue

                n_seq = self._count_sequences_for_recording(
                    label_path, spatial_factor, seq_length, stride,
                    temp_subsample_factor
                )
                if n_seq > 0:
                    valid_rows.append((row, n_seq))
                    total_sequences += n_seq

            if total_sequences == 0:
                print(f"  Warning: No features extracted for {split}")
                continue

            print(f"  Total sequences: {total_sequences}")

            # === Pass 2: Create memmaps and fill incrementally ===
            print(f"  Pass 2: Extracting and writing to memmap...")
            feat_shape = (total_sequences, seq_length, n_time_bins, ds_height, ds_width)
            tgt_shape = (total_sequences, seq_length, 3)

            feat_path = os.path.join(output_dir, 'features.npy')
            tgt_path = os.path.join(output_dir, 'targets.npy')

            # Create pre-allocated memmap files
            feat_mm = np.lib.format.open_memmap(
                feat_path, mode='w+', dtype=np.float32, shape=feat_shape
            )
            tgt_mm = np.lib.format.open_memmap(
                tgt_path, mode='w+', dtype=np.float32, shape=tgt_shape
            )

            offset = 0
            for row, n_seq in tqdm(valid_rows, desc=f'Extract {split}'):
                h5_path = row['h5_path']
                label_path = row['label_path']

                try:
                    features, targets = self._process_recording(
                        h5_path, label_path, spatial_factor,
                        n_time_bins, seq_length, stride,
                        sensor_width, sensor_height,
                        ds_width, ds_height,
                        temp_subsample_factor
                    )
                    if features is not None and len(features) > 0:
                        n_new = features.shape[0]
                        feat_mm[offset:offset + n_new] = features
                        tgt_mm[offset:offset + n_new] = targets
                        offset += n_new
                        del features, targets
                except Exception as e:
                    print(f"  Error processing {row['recording_id']}: {e}")
                    continue

            # Flush to disk
            del feat_mm, tgt_mm

            # Save metadata
            metadata = {
                'n_samples': offset,
                'n_features': n_time_bins,
                'n_classes': 2,
                'sensor_size': [ds_width, ds_height],
                'seq_length': seq_length,
                'shape_features': list(feat_shape),
                'shape_targets': list(tgt_shape),
            }
            meta_path = os.path.join(output_dir, 'metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  Saved to {output_dir}")
            print(f"  Features shape: ({offset}, {seq_length}, {n_time_bins}, {ds_height}, {ds_width})")
            print(f"  Targets shape: ({offset}, {seq_length}, 3)")
            if offset < total_sequences:
                print(f"  Note: {total_sequences - offset} sequences skipped due to errors")

        print("\nFeature stored in: ", os.path.join('feat', args.dataset_name))
        print("Feature Extraction Completed...")

    def _count_sequences_for_recording(self, label_path, spatial_factor,
                                        seq_length, stride,
                                        temp_subsample_factor):
        """Count number of sequences a recording will produce (lightweight, no event loading)."""
        if label_path is None or not os.path.exists(label_path):
            return 0

        # Count label lines
        n_labels = 0
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace('(', '').replace(')', '').replace(',', ' ').split()
                if len(parts) >= 2:
                    n_labels += 1

        if n_labels == 0:
            return 0

        # Apply temporal subsampling
        if temp_subsample_factor < 1.0:
            interval = int(1.0 / temp_subsample_factor)
            n_frames = len(range(0, n_labels, interval))
        else:
            n_frames = n_labels

        if n_frames < seq_length:
            return 0

        # Number of sequences with given stride
        n_seq = (n_frames - seq_length) // stride + 1
        return n_seq

    def _process_recording(self, h5_path, label_path, spatial_factor,
                           n_time_bins, seq_length, stride,
                           sensor_width, sensor_height,
                           ds_width, ds_height,
                           temp_subsample_factor):
        """Process a single recording: events -> voxel grid sequences."""
        # Load events from H5
        with h5py.File(h5_path, 'r') as hf:
            events_t = np.array(hf['t'], dtype=np.float64)
            events_x = np.array(hf['x'], dtype=np.float64)
            events_y = np.array(hf['y'], dtype=np.float64)
            events_p = np.array(hf['p'], dtype=np.float64)

        if len(events_t) == 0:
            return None, None

        # Apply spatial downsampling to events
        events_x = events_x * spatial_factor
        events_y = events_y * spatial_factor

        # Convert polarity: 0/1 -> -1/+1
        events_p = events_p * 2 - 1

        # Load labels
        labels = self._load_labels(label_path, spatial_factor,
                                   sensor_width, sensor_height,
                                   ds_width, ds_height)
        if labels is None or len(labels) == 0:
            return None, None

        # Temporal subsampling
        if temp_subsample_factor < 1.0:
            interval = int(1.0 / temp_subsample_factor)
            labels = labels[::interval]

        n_frames = len(labels)
        if n_frames < seq_length:
            return None, None

        # Slice events into time windows aligned with labels (100Hz = 10ms per frame)
        t_start = events_t[0]
        t_end = events_t[-1]
        frame_duration = (t_end - t_start) / n_frames

        # Build per-frame voxel grids
        voxel_grids = []
        for i in range(n_frames):
            frame_t_start = t_start + i * frame_duration
            frame_t_end = t_start + (i + 1) * frame_duration

            # Get events in this time window
            mask = (events_t >= frame_t_start) & (events_t < frame_t_end)
            frame_x = events_x[mask]
            frame_y = events_y[mask]
            frame_t = events_t[mask]
            frame_p = events_p[mask]

            # Clip coordinates to valid range
            frame_x = np.clip(frame_x, 0, ds_width - 1)
            frame_y = np.clip(frame_y, 0, ds_height - 1)

            # Convert to voxel grid
            vg = custom_to_voxel_grid_numpy(
                frame_x, frame_y, frame_t, frame_p,
                ds_width, ds_height, n_time_bins
            )
            voxel_grids.append(vg)  # (C, H, W)

        voxel_grids = np.stack(voxel_grids, axis=0)  # (n_frames, C, H, W)

        # Split into sequences
        features_list = []
        targets_list = []
        for i in range(0, n_frames - seq_length + 1, stride):
            seq_features = voxel_grids[i:i + seq_length]    # (T, C, H, W)
            seq_targets = labels[i:i + seq_length]           # (T, 3)
            features_list.append(seq_features)
            targets_list.append(seq_targets)

        if len(features_list) == 0:
            return None, None

        features = np.stack(features_list, axis=0)  # (N_seq, T, C, H, W)
        targets = np.stack(targets_list, axis=0)     # (N_seq, T, 3)

        return features, targets

    def _load_labels(self, label_path, spatial_factor,
                     sensor_width, sensor_height,
                     ds_width, ds_height):
        """Load and process labels from label.txt.

        Returns:
            numpy array of shape (n_frames, 3) with [x_norm, y_norm, close_flag]
            where x_norm and y_norm are in [0, 1] range.
        """
        if label_path is None or not os.path.exists(label_path):
            return None

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse label: each line is "x y close" or "(x, y, close)"
                line = line.replace('(', '').replace(')', '').replace(',', ' ')
                parts = line.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    close = float(parts[2]) if len(parts) >= 3 else 0.0

                    # Scale by spatial factor
                    x *= spatial_factor
                    y *= spatial_factor

                    # Normalize to [0, 1]
                    x_norm = x / ds_width
                    y_norm = y / ds_height

                    labels.append([x_norm, y_norm, close])

        if len(labels) == 0:
            return None

        return np.array(labels, dtype=np.float32)
