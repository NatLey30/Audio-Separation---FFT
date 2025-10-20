import torch
import torch.nn as nn
import numpy as np
import librosa


# === FFT/STFT Frontend ===
class STFTFrontend(nn.Module):
    def __init__(self, n_fft=1024, hop=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, x):  # x: (B,1,T)
        # Convertir a numpy
        x_np = x.squeeze(1).cpu().numpy()
        feats, phases = [], []
        for i in range(x_np.shape[0]):
            D = librosa.stft(x_np[i], n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length)
            feats.append(np.stack([np.real(D), np.imag(D)], axis=0))
            phases.append(D)
        feat = torch.tensor(np.stack(feats), dtype=torch.float32)  # (B,2,F,Tf)
        Xc = torch.tensor(np.stack(phases), dtype=torch.complex64) # (B,F,Tf)
        return feat, Xc

    def istft(self, Xc, length=None):
        Xc_np = Xc.detach().cpu().numpy()
        outs = []
        for i in range(Xc_np.shape[0]):
            y = librosa.istft(Xc_np[i], hop_length=self.hop, win_length=self.win_length, length=length)
            outs.append(torch.tensor(y, dtype=torch.float32))
        return torch.stack(outs).unsqueeze(1)  # (B,1,T)


# === Conv aprendible Frontend ===
class ConvFrontend(nn.Module):
    def __init__(self, n_filters=512, kernel=40, stride=20):
        super().__init__()
        self.enc = nn.Conv1d(1, n_filters, kernel_size=kernel, stride=stride, bias=False)
        self.dec = nn.ConvTranspose1d(n_filters, 1, kernel_size=kernel, stride=stride, bias=False)

    def forward(self, x):               # x: (B,1,T)
        Z = self.enc(x)                 # (B,C,L)
        return Z, None

    def decode(self, Z):                # (B,C,L) -> (B,1,T)
        return self.dec(Z)
