"""
Data collector for Google Speech Commands Dataset v2.
Implements 12-class keyword spotting:
- 10 keywords: yes, no, up, down, left, right, on, off, stop, go
- _silence_: background noise
- _unknown_: all other words
"""
import os
import pandas as pd
from tqdm import tqdm
import random
from glob import glob


# 12-class KWS labels
KWS_KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
LABEL_SILENCE = '_silence_'
LABEL_UNKNOWN = '_unknown_'
KWS_LABELS = [LABEL_SILENCE, LABEL_UNKNOWN] + KWS_KEYWORDS  # 12 classes total


class DataPrepare:
    def __init__(self, dataset_path, output_path, sample_rate=16000, audio_length=1.0):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.target_length = int(sample_rate * audio_length)
        os.makedirs(output_path, exist_ok=True)

        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(KWS_LABELS)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Save label list
        label_df = pd.DataFrame({
            'label': KWS_LABELS,
            'idx': range(len(KWS_LABELS))
        })
        label_df.to_csv(os.path.join(output_path, 'label_list.csv'), index=False)
        print(f"Labels: {KWS_LABELS}")
        print(f"Saved label list to {os.path.join(output_path, 'label_list.csv')}")

    def _get_label(self, word):
        """Map word to 12-class KWS label index."""
        if word in KWS_KEYWORDS:
            return self.label_to_idx[word]
        elif word == '_background_noise_':
            return self.label_to_idx[LABEL_SILENCE]
        else:
            return self.label_to_idx[LABEL_UNKNOWN]

    def collect(self, **kwargs):
        """Collect and prepare the dataset from existing files."""
        print("Loading Google Speech Commands v2 dataset from disk...")

        # Find the dataset directory
        gscd_dir = None

        # Case 1: dataset_path already points to the dataset root
        direct_val = os.path.join(self.dataset_path, 'validation_list.txt')
        direct_test = os.path.join(self.dataset_path, 'testing_list.txt')
        if os.path.exists(direct_val) and os.path.exists(direct_test):
            gscd_dir = self.dataset_path
        else:
            # Case 2: SpeechCommands/speech_commands_v0.02
            candidate = os.path.join(self.dataset_path, 'SpeechCommands', 'speech_commands_v0.02')
            if os.path.exists(candidate):
                gscd_dir = candidate
            else:
                # Case 3: speech_commands_v0.02 under dataset_path
                candidate = os.path.join(self.dataset_path, 'speech_commands_v0.02')
                if os.path.exists(candidate):
                    gscd_dir = candidate

        if gscd_dir is None:
            raise RuntimeError(
                "Dataset not found. Expected one of:\n"
                f"  - {self.dataset_path} (containing validation_list.txt/testing_list.txt)\n"
                f"  - {os.path.join(self.dataset_path, 'SpeechCommands', 'speech_commands_v0.02')}\n"
                f"  - {os.path.join(self.dataset_path, 'speech_commands_v0.02')}\n"
                "Please download the dataset first."
            )

        print(f"Found dataset at: {gscd_dir}")

        # Load validation and testing lists
        val_list_file = os.path.join(gscd_dir, 'validation_list.txt')
        test_list_file = os.path.join(gscd_dir, 'testing_list.txt')

        with open(val_list_file, 'r') as f:
            val_files = set(f.read().strip().split('\n'))
        with open(test_list_file, 'r') as f:
            test_files = set(f.read().strip().split('\n'))

        # Collect all audio files
        train_data = []
        val_data = []
        test_data = []

        # Get all word directories
        word_dirs = [d for d in os.listdir(gscd_dir)
                     if os.path.isdir(os.path.join(gscd_dir, d)) and not d.startswith('_')]

        print(f"Found {len(word_dirs)} word categories")

        for word in tqdm(word_dirs, desc='Processing words'):
            word_dir = os.path.join(gscd_dir, word)
            wav_files = glob(os.path.join(word_dir, '*.wav'))

            label_idx = self._get_label(word)
            label_name = self.idx_to_label[label_idx]

            for wav_file in wav_files:
                # Get relative path for checking val/test lists
                rel_path = f"{word}/{os.path.basename(wav_file)}"

                sample = {
                    'group': 'train',
                    'path': wav_file,
                    'label': label_name,
                    'label_idx': label_idx,
                    'word': word
                }

                if rel_path in val_files:
                    sample['group'] = 'val'
                    val_data.append(sample)
                elif rel_path in test_files:
                    sample['group'] = 'test'
                    test_data.append(sample)
                else:
                    sample['group'] = 'train'
                    train_data.append(sample)

        # Add silence samples from background noise
        train_data = self._add_silence_samples(train_data, gscd_dir, 'train', n_samples=2000)
        val_data = self._add_silence_samples(val_data, gscd_dir, 'val', n_samples=250)
        test_data = self._add_silence_samples(test_data, gscd_dir, 'test', n_samples=250)

        # Save to CSV description files
        columns = ['group', 'path', 'label', 'label_idx', 'word']

        train_df = pd.DataFrame(train_data, columns=columns)
        val_df = pd.DataFrame(val_data, columns=columns)
        test_df = pd.DataFrame(test_data, columns=columns)

        train_df.to_csv(os.path.join(self.output_path, 'train_description.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_path, 'val_description.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_path, 'test_description.csv'), index=False)

        print(f"\nDataset statistics:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Val: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")

        # Print class distribution
        self._print_class_distribution(train_df, 'Train')
        self._print_class_distribution(val_df, 'Val')
        self._print_class_distribution(test_df, 'Test')

    def _add_silence_samples(self, data, gscd_dir, group, n_samples=1000):
        """Add silence samples from background noise files."""
        noise_dir = os.path.join(gscd_dir, '_background_noise_')

        if not os.path.exists(noise_dir):
            print(f"Warning: Background noise directory not found at {noise_dir}")
            return data

        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

        if not noise_files:
            print("Warning: No background noise files found")
            return data

        silence_samples = []
        for i in range(n_samples):
            noise_file = random.choice(noise_files)
            filepath = os.path.join(noise_dir, noise_file)

            silence_samples.append({
                'group': group,
                'path': filepath,
                'label': LABEL_SILENCE,
                'label_idx': self.label_to_idx[LABEL_SILENCE],
                'word': '_background_noise_'
            })

        data.extend(silence_samples)
        return data

    def _print_class_distribution(self, df, split_name):
        """Print class distribution."""
        print(f"\n{split_name} class distribution:")
        for label in KWS_LABELS:
            count = len(df[df['label'] == label])
            print(f"  {label}: {count}")
