import librosa
import numpy as np
from add_noise import add_noise
import torch
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean, _ = self.data[idx]
        return torch.tensor(noisy)[None, :], torch.tensor(clean)[None, :]


def make_dataset(snr_db=5, duration=5):
    examples = ['trumpet', 'nutcracker', 'brahms', 'fishin', 'choice']
    data = []
    for ex in examples:
        try:
            y, sr = librosa.load(librosa.example(ex), sr=None, mono=True, duration=duration)
            y = y / (np.max(np.abs(y)) + 1e-8)
            y_noisy = add_noise(y, snr_db=snr_db)
            data.append((y_noisy, y, sr))
        except Exception as e:
            print(f"No se pudo cargar ejemplo {ex}: {e}")
    return data
