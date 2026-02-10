"""
Feature extractor for Google Speech Commands Dataset v2.
Extracts log mel-filterbank features from audio files.
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.io import wavfile
from tqdm import tqdm
from utils.feature.filter_bank import compute_filter_bank
from modules.gscdv2 import log


class FeatExtractor:
    def __init__(self):
        self.description_name_train = 'train_description.csv'
        self.description_name_val = 'val_description.csv'
        self.description_name_test = 'test_description.csv'

    def extract(self, args):
        """Extract features for all dataset splits."""
        data_dir = os.path.join('data', args.dataset_name)

        # Get output file names
        _, trainfile, valfile, testfile = log.gen_dataset_name(args)

        # Parameters
        sample_rate = getattr(args, 'sample_rate', 16000)
        audio_length = getattr(args, 'audio_length', 1.0)
        target_samples = int(sample_rate * audio_length)

        for split, desc_file, output_file in [
            ('train', self.description_name_train, trainfile),
            ('val', self.description_name_val, valfile),
            ('test', self.description_name_test, testfile)
        ]:
            print(f"\nExtracting features for {split} set...")

            # Load description file
            desc_path = os.path.join(data_dir, desc_file)
            df = pd.read_csv(desc_path)

            features_list = []
            labels_list = []

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Extract {split}'):
                is_silence = row['label'] == '_silence_'

                try:
                    waveform = self._load_and_preprocess_audio(
                        row['path'], sample_rate, target_samples, is_silence
                    )
                    features = self._extract_features(waveform, args, sample_rate)
                    features_list.append(features)
                    labels_list.append(row['label_idx'])
                except Exception as e:
                    print(f"Error processing {row['path']}: {e}")
                    continue

            # Stack all features
            all_features = np.stack(features_list, axis=0)  # (N, T, F)
            all_labels = np.array(labels_list, dtype=np.int32)

            # Get dimensions
            n_features = all_features.shape[2]
            n_classes = getattr(args, 'n_classes', 12)

            # Save to HDF5
            output_path = os.path.join('feat', args.dataset_name, output_file)
            print(f"Saving to {output_path}")
            print(f"  Shape: {all_features.shape}")
            print(f"  n_features: {n_features}, n_classes: {n_classes}")

            with h5py.File(output_path, 'w') as hf:
                hf.create_dataset('features', data=all_features)
                hf.create_dataset('labels', data=all_labels)
                hf.create_dataset('n_features', data=n_features)
                hf.create_dataset('n_classes', data=n_classes)

        print("\nFeature stored in: ", os.path.join('feat', args.dataset_name))
        print("Feature Extraction Completed...")

    def _load_and_preprocess_audio(self, filepath, sample_rate, target_samples, is_silence=False):
        """Load and preprocess audio file using scipy."""
        sr, waveform = wavfile.read(filepath)

        # Convert to float and normalize
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        elif waveform.dtype == np.int32:
            waveform = waveform.astype(np.float32) / 2147483648.0
        else:
            waveform = waveform.astype(np.float32)

        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        # For silence samples, extract a random segment
        if is_silence and len(waveform) > target_samples:
            start = np.random.randint(0, len(waveform) - target_samples)
            waveform = waveform[start:start + target_samples]

        # Pad or trim to target length
        if len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)))
        elif len(waveform) > target_samples:
            waveform = waveform[:target_samples]

        return waveform

    def _extract_features(self, waveform, args, sample_rate):
        """Extract log mel-filterbank features."""
        # Compute filter bank features (already includes gradient if requested)
        features = compute_filter_bank(
            signal=waveform,
            sample_rate=sample_rate,
            frame_size=args.frame_size,
            frame_stride=args.frame_stride,
            NFFT=args.nfft,
            n_filt=args.n_filt,
            gain_ctrl=args.gain_ctrl,
            gradient=args.gradient
        )

        return features.astype(np.float32)
