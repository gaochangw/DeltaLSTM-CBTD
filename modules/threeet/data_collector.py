"""
Data collector for 3ET+ Event-Based Eye Tracking dataset.
Verifies data exists and creates description CSV cataloging all recordings.
"""
import os
import pandas as pd


class DataPrepare:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def collect(self, **kwargs):
        """Verify 3ET+ data exists and create description CSV."""
        print("Preparing 3ET+ Event-Based Eye Tracking dataset...")

        if self.dataset_path is None:
            raise RuntimeError(
                "Please provide the path to the 3ET+ dataset via --data_dir."
            )

        if not os.path.exists(self.dataset_path):
            raise RuntimeError(f"Dataset path does not exist: {self.dataset_path}")

        # Load split files
        split_files = {}
        for split in ['train', 'val', 'test']:
            split_file = os.path.join('data', 'threeet', f'{split}_files.txt')
            if not os.path.exists(split_file):
                raise RuntimeError(
                    f"Split file not found: {split_file}\n"
                    f"Please ensure data/threeet/{split}_files.txt exists."
                )
            with open(split_file, 'r') as f:
                split_files[split] = [line.strip() for line in f if line.strip()]

        # Catalog all recordings
        records = []
        for split, recording_ids in split_files.items():
            for rec_id in recording_ids:
                # Expected path: {data_dir}/train/{subject}_{session}/{subject}_{session}.h5
                # or: {data_dir}/{subject}_{session}/{subject}_{session}.h5
                h5_path = None
                label_path = None

                # Try multiple directory structures
                candidates = [
                    os.path.join(self.dataset_path, 'train', rec_id, f'{rec_id}.h5'),
                    os.path.join(self.dataset_path, rec_id, f'{rec_id}.h5'),
                    os.path.join(self.dataset_path, 'data', rec_id, f'{rec_id}.h5'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        h5_path = candidate
                        label_path = os.path.join(os.path.dirname(candidate), 'label.txt')
                        break

                if h5_path is None:
                    print(f"  Warning: H5 file not found for {rec_id}, tried:")
                    for c in candidates:
                        print(f"    {c}")
                    continue

                records.append({
                    'group': split,
                    'recording_id': rec_id,
                    'h5_path': h5_path,
                    'label_path': label_path,
                    'label_exists': os.path.exists(label_path) if label_path else False,
                })

        # Save description CSV
        df = pd.DataFrame(records)
        desc_path = os.path.join(self.output_path, 'description.csv')
        df.to_csv(desc_path, index=False)

        # Print summary
        print(f"\n3ET+ Dataset Summary:")
        print(f"  Description saved to: {desc_path}")
        for split in ['train', 'val', 'test']:
            split_df = df[df['group'] == split]
            n_found = len(split_df)
            n_expected = len(split_files[split])
            print(f"  {split}: {n_found}/{n_expected} recordings found")

        n_missing = len(df[~df['label_exists']]) if 'label_exists' in df.columns else 0
        if n_missing > 0:
            print(f"  Warning: {n_missing} recordings missing label.txt")
