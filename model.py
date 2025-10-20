import torch
import torch.nn as nn
from frontends import STFTFrontend, ConvFrontend
from unet_separator import UNet2D


class ToggleSeparator(nn.Module):
    def __init__(self, mode="stft", n_sources=2):
        super().__init__()
        assert mode in ("stft","conv")
        self.mode = mode
        self.n_sources = n_sources

        if mode == "stft":
            self.fe = STFTFrontend()
            self.unet = UNet2D(in_ch=2, out_ch=2*n_sources)
        else:
            self.fe = ConvFrontend()
            self.unet = UNet2D(in_ch=1, out_ch=n_sources)  # U-Net 1D adaptada

    def forward(self, x):  # x: (B,1,T)
        if self.mode == "stft":
            feat, Xc = self.fe(x)
            out = self.unet(feat)
            masks = out.view(x.size(0), self.n_sources, 2, *feat.shape[2:])
            X = torch.complex(Xc.real, Xc.imag).unsqueeze(1).expand(-1,self.n_sources,-1,-1)
            M = torch.complex(masks[:,:,0], masks[:,:,1])
            Shat = M * X
            outs = [self.fe.istft(Shat[:,s], length=x.size(-1)) for s in range(self.n_sources)]
            return torch.stack(outs, dim=1).squeeze(2)
        else:
            # Z, _ = self.fe(x)              # (B,C,L)
            # Z = Z.unsqueeze(1)             # (B,1,C,L)
            # out = self.unet(Z)
            # yhat = torch.sigmoid(out) * x  # simplificación
            # return yhat.unsqueeze(1)
            # --- Codificación ---
            Z, _ = self.fe(x)               # (B, C, L)
            Z = Z.unsqueeze(1)              # (B, 1, C, L)
            out = self.unet(Z)              # (B, n_sources, C', L')

            # --- Proyección 1x1 para colapsar canales (C' -> 1) ---
            B, S, C, Lp = out.shape
            proj = torch.nn.Conv1d(C, 1, kernel_size=1).to(out.device)
            out_reshaped = out.view(B * S, C, Lp)       # (B*S, C, L')
            out_wave = proj(out_reshaped)               # (B*S, 1, L')
            out_wave = out_wave.view(B, S, Lp)          # (B, S, L')

            # --- Activación e interpolación ---
            out_wave = torch.sigmoid(out_wave)
            out_wave = torch.nn.functional.interpolate(
                out_wave, size=x.shape[-1], mode="linear", align_corners=False
            )  # (B, S, T)

            # --- Escalamos por la señal original ---
            yhat = out_wave * x  # broadcasting: (B, S, T)
            return yhat
