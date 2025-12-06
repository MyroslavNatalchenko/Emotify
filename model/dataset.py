import torch
from torch.utils.data import Dataset
import numpy as np

def load_and_process_spectrogram(path, target_length=1366, n_mels=96):
    """
    Loading .npy file, creating Center Crop or Padding до target_length of melspectrogram
    Returning: numpy array (n_mels, target_length)
    """
    try:
        spec = np.load(path)  # shape: (n_mels, time)
    except Exception as e:
        return np.zeros((n_mels, target_length), dtype=np.float32)

    current_len = spec.shape[1]

    # CENTER CROP
    if current_len > target_length:
        center = current_len // 2
        half_target = target_length // 2
        start = center - half_target
        end = start + target_length

        if start < 0: start = 0; end = target_length
        if end > current_len: end = current_len; start = end - target_length

        spec = spec[:, start:end]
    elif current_len < target_length:
        pad_amount = target_length - current_len
        spec = np.pad(spec, ((0, 0), (0, pad_amount)), mode='constant')

    return spec.astype(np.float32)


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

        spec = load_and_process_spectrogram(npy_path, self.target_length)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0)

        labels = torch.tensor(row[self.label_cols].values.astype('float32'))

        return spec_tensor, labels