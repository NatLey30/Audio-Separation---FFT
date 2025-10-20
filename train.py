import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
import numpy as np

from model import ToggleSeparator
from data import make_dataset, NoiseDataset
from train_functions import train_one_epoch


# === Full training function ===
def train_model(mode="stft", epochs=5, snr_db=5, batch_size=1, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando modelo '{mode}' en {device}")

    data = make_dataset(snr_db=snr_db)
    train_loader = DataLoader(NoiseDataset(data), batch_size=batch_size, shuffle=True)

    model = ToggleSeparator(mode=mode, n_sources=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"[{mode}] Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, f"unet_{mode}_denoise.pth"))
    print(f"Modelo '{mode}' guardado en {out_dir}")
    return model, out_dir


# === Train both models (STFT + CONV) ===
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # parámetros
    EPOCHS = 10
    SNR_DB = 5
    BATCH_SIZE = 1

    # entrena ambas unets
    model_stft, dir_stft = train_model("stft", epochs=EPOCHS, snr_db=SNR_DB, batch_size=BATCH_SIZE)
    model_conv, dir_conv = train_model("conv", epochs=EPOCHS, snr_db=SNR_DB, batch_size=BATCH_SIZE)

    print("\n Entrenamiento finalizado.")
    print(f"Modelos guardados en:\n  STFT → {dir_stft}\n  CONV → {dir_conv}")
