# Audio Separation using FFT vs Convolutional U-Net

This project explores **two different front-end representations** for audio source separation and denoising:

- **STFT-based U-Net** — uses the Short-Time Fourier Transform to learn masks in the time-frequency domain.  
- **Conv-based U-Net** — operates directly on raw waveforms with learned 1D convolutions.

The goal is to compare how the explicit Fourier representation (magnitude + phase) affects reconstruction quality compared to a purely temporal convolutional approach.

---

## Project Overview

| Component | Description |
|------------|-------------|
| **STFTFrontend** | Computes the complex STFT of the input waveform. Returns magnitude and phase for the network. |
| **ConvFrontend** | Learns its own analysis filters directly from the waveform using Conv1D. |
| **UNet2D** | Standard encoder–decoder with skip connections (Conv2D or Conv1D, depending on the mode). |
| **ToggleSeparator** | Wrapper that switches between `stft` and `conv` modes for fair comparison. |
| **Training** | Both models are trained to **denoise** synthetic mixtures (clean + noise at 5 dB SNR). |
| **Evaluation** | Metrics such as SI-SDR and SNR Improvement are used to assess performance. |

---

## Repository Structure

```text
Audio-Separation---FFT/
 │
 ├── frontends.py # STFT and Conv1D feature extractors
 ├── unet_separator.py # U-Net architecture (encoder–decoder)
 ├── model.py # ToggleSeparator: wraps frontend + U-Net
 ├── add_noise.py # Adds Gaussian noise at chosen SNR
 ├── metrics.py # SI-SDR, SNRi, (optionally LSD/STOI/PESQ)
 ├── train_functions.py # Training loop utilities (device-safe)
 ├── train.py # Trains STFT and Conv U-Nets (denoising)
 ├── main.py # Inference + metrics on a noisy sample
 ├── runs/ # Auto-created folders for logs/outputs
 └── models/ # Saved weights (.pth)
```

---

## Requirements

```bash
# PyTorch with CUDA 12.4 (recommended for RTX 50xx)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core Python deps
pip install librosa soundfile numpy matplotlib
```

## Training

Train both models (STFT and Conv) on synthetic denoising (5 dB SNR):

```bash
python train.py
```

The models are stored in:
```text
models/
 ├── unet_stft_denoise.pth # STFT-based U-Net (frequency domain)
 └── unet_conv_denoise.pth # Conv1D-based U-Net (time domain)
```


## Evaluation and Inference

Compare both models on a noisy input:

```bash
python main.py
```

Outputs are stored in a timestamped folder:
```text
runs/comparison_YYYY-MM-DD_HH-MM-SS/
 ├── input_clean.wav
 ├── input_noisy_5dB.wav
 ├── output_stft.wav
 ├── output_conv.wav
 └── results.txt
```

## Example results:

```bash
Noise added: 5 dB SNR

[STFT]
SI_SDR: 3.87
SNRi: 2.11

[CONV]
SI_SDR: 1.05
SNRi: 0.45
```

## Experiments & Observations

- **Data:** Librosa demo clips (`trumpet`, `brahms`, `nutcracker`, `fishin`, `choice`).
- **Task:** Single-channel denoising (clean target, noisy input).
- **Noise:** Gaussian, 5 dB SNR.
- **Frontends compared:**
  - **STFT U-Net:** Learns complex masks; reconstructs with inverse STFT.
  - **Conv U-Net:** Learns waveform-level features with Conv1D filters.
- **Observation:** STFT-based U-Net converges faster and performs better on stationary noise, while the Conv1D version tends to handle transient artifacts better with more data.

## Author

Project by **Natalia Leyenda**  & **Sofía Pedrós**

