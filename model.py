import torch
import torch.nn as nn
from frontends import STFTFrontend, ConvFrontend
from unet_separator import UNet2D


class ToggleSeparator(nn.Module):
    def __init__(self, mode="stft", n_sources=2, device=None):
        super().__init__()
        assert mode in ("stft", "conv")
        self.mode = mode
        self.n_sources = n_sources

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "stft":
            self.fe = STFTFrontend().to(self.device)
            self.unet = UNet2D(in_ch=2, out_ch=2 * n_sources).to(self.device)
        else:
            self.fe = ConvFrontend().to(self.device)
            self.unet = UNet2D(in_ch=1, out_ch=n_sources).to(self.device)

    def forward(self, x):  # x: (B,1,T)
        x = x.to(self.device)

        if self.mode == "stft":
            self.fe.to(self.device)
            feat, Xc = self.fe(x)
            feat, Xc = feat.to(self.device), Xc.to(self.device)

            out = self.unet(feat)
            masks = out.view(x.size(0), self.n_sources, 2, *feat.shape[2:])
            X = torch.complex(Xc.real, Xc.imag).unsqueeze(1).expand(-1, self.n_sources, -1, -1)
            M = torch.complex(masks[:, :, 0], masks[:, :, 1])
            Shat = M * X

            Shat = Shat.to(x.device)
            outs = [torch.istft(Shat[:, s], n_fft=1024, hop_length=256, length=x.size(-1)) 
                    for s in range(self.n_sources)]
            return torch.stack(outs, dim=1).squeeze(2)

        else:
            self.fe.to(self.device)
            Z, _ = self.fe(x)
            Z = Z.to(self.device).unsqueeze(1)
            out = self.unet(Z)

            # --- Proyección 1x1 ---
            B, S, C, Lp = out.shape
            proj = torch.nn.Conv1d(C, 1, kernel_size=1).to(self.device)
            out_reshaped = out.view(B * S, C, Lp)
            out_wave = proj(out_reshaped)
            out_wave = out_wave.view(B, S, Lp)

            out_wave = torch.sigmoid(out_wave)
            out_wave = torch.nn.functional.interpolate(
                out_wave, size=x.shape[-1], mode="linear", align_corners=False
            )

            yhat = out_wave * x
            return yhat
