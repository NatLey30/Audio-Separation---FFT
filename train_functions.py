import torch
from metrics import batch_si_sdr
import numpy as np


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        print(noisy.shape)
        out = model(noisy)     # (B,1,T)
        # Para n_sources=1 → out.shape = (B,1,T)
        loss = loss_fn(out[:, 0, :], clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    model.eval()
    all_sisdr = []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy.unsqueeze(1))
            sisdr = batch_si_sdr(out, clean.unsqueeze(1).unsqueeze(1))
            all_sisdr.append(float(sisdr.mean()))
    return np.mean(all_sisdr)
