import torch
import torch.nn as nn
import torch.nn.functional as F


# === Bloque básico Conv2d ===
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# === U-Net espectral ===
class UNet2D(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=64, depth=4):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = base_ch
        for _ in range(depth):
            self.downs.append(ConvBlock2D(in_ch, ch))
            in_ch = ch
            self.pools.append(nn.MaxPool2d(2))
            ch *= 2

        self.bottom = ConvBlock2D(in_ch, ch)

        for _ in range(depth):
            ch //= 2
            self.ups.append(nn.ConvTranspose2d(ch*2, ch, kernel_size=2, stride=2))
            self.ups.append(ConvBlock2D(ch*2, ch))

        self.final = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skips.append(x)
            x = self.pools[i](x)
        x = self.bottom(x)
        for i in range(0, len(self.ups), 2):
            up_trans = self.ups[i]
            up_conv = self.ups[i+1]
            x = up_trans(x)
            skip = skips[-(i//2+1)]
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)
        return self.final(x)
