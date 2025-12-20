import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class MultimodalDataset(Dataset):
    def __init__(self, df, mert_emb_dir, target_length=1366):
        """
        Args:
            df: DataFrame containing the data.
            mert_emb_dir: Path to the directory containing MERT npy files.
            target_length: Spectrogram width.
        """
        self.df = df
        self.mert_emb_dir = mert_emb_dir
        self.target_length = target_length

        self.meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
        self.label_cols = [col for col in self.df.columns if col not in self.meta_cols]

        # Caching MERT vectors [for fast access]
        print(f"Caching MERT embeddings ({len(df)} files)...")
        self.mert_vectors = []
        self.labels = []

        # Pre-calculate paths
        paths = self.df['PATH'].values
        labels_matrix = self.df[self.label_cols].values.astype('float32')

        for idx in tqdm(range(len(self.df))):
            original_path = paths[idx]

            filename = os.path.basename(original_path)
            folder = os.path.basename(os.path.dirname(original_path))

            # Filename candidates [to rule out all possible errors]
            candidates = [
                filename.replace('.mp3', '.npy'),
                filename.replace('.mp3', '.npy').replace('.low.npy', '.npy'),
                filename.replace('.mp3', '.npy').replace('.npy', '.low.npy')
            ]
            if not filename.endswith('.mp3'):
                candidates.append(filename)

            emb_path = None
            for cand in candidates:
                full_path = os.path.join(self.mert_emb_dir, folder, cand)
                if os.path.exists(full_path):
                    emb_path = full_path
                    break

            # Load MERT
            if emb_path:
                try:
                    vec = np.load(emb_path)
                    if vec.ndim > 1: vec = vec.mean(axis=0)  # (Time, 768) -> (768,)
                    if vec.shape[0] != 768: vec = np.zeros(768)  # Fallback
                    self.mert_vectors.append(vec)
                except:
                    self.mert_vectors.append(np.zeros(768))
            else:
                self.mert_vectors.append(np.zeros(768))

            self.labels.append(labels_matrix[idx])

        # Convert MERT and Labels to tensors immediately
        self.mert_vectors = torch.tensor(np.array(self.mert_vectors), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spec_path = row['PATH']
        try:
            spec = np.load(spec_path).astype(np.float32)
            spec_tensor = torch.from_numpy(spec)
        except:
            spec_tensor = torch.zeros((96, self.target_length))
            print(f"Created zero tensor for npy {spec_path} \n")

        mert_tensor = self.mert_vectors[idx]

        label_tensor = self.labels[idx]

        return spec_tensor, mert_tensor, label_tensor