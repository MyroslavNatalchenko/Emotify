import torch
from torch.utils.data import Dataset
import numpy as np

class AudioMoodDataset(Dataset):
    def __init__(self, df, target_length=1366):
        self.df = df
        self.target_length = target_length
        self.label_cols = df.columns[3:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = row['PATH']

        spec = np.load(npy_path).astype(np.float32)

        spec_tensor = torch.from_numpy(spec)

        labels = torch.tensor(row[self.label_cols].values.astype('float32'))

        return spec_tensor, labels