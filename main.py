import os
from datetime import datetime
import numpy as np
import torch
import librosa
import soundfile as sf

from model import ToggleSeparator
from add_noise import add_noise
from metrics import batch_si_sdr, snri_db


def run_mode(mode, output_dir, wav, sr, mixture_noisy, target):
    """Ejecuta un modo (stft o conv) y guarda resultados."""
    # --- Cargar modelo entrenado ---
    model = ToggleSeparator(mode=mode, n_sources=2)
    ckpt_path = f"models/unet_{mode}_denoise.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    print(f"Modelo {mode} cargado desde {ckpt_path}")


    print(f"Ejecutando modo: {mode}")
    with torch.no_grad():
        separated = model(wav)

    separated_np = separated.squeeze(0).cpu().numpy()  # (2,T)

    # Guardar resultados
    for i in range(separated.shape[1]):
        sf.write(os.path.join(output_dir, f"output_{mode}_source{i+1}.wav"),
                 separated[0, i].cpu().numpy(), sr)
    print(f"Resultados guardados para modo {mode}")

    # === Calcular métricas (solo respecto a la señal original limpia) ===
    # target shape (T,)
    est1 = separated_np[0]
    si_sdr_val = batch_si_sdr(torch.tensor(separated_np)[None, ...],
                              torch.tensor(np.stack([target, target])[None, ...]))
    snri_val = snri_db(mixture_noisy, est1, target)

    return {
        "SI_SDR_source1": float(si_sdr_val[0, 0]),
        "SNRi_source1": float(snri_val),
    }


def demo():
    # === Crear carpeta común ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("runs", f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando resultados en: {output_dir}")

    # === Cargar audio de ejemplo ===
    audio_path = librosa.example('trumpet') # nutcracker, brahms, fishin
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    # wav = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # (1,1,T)

    y = y / np.max(np.abs(y))  # normaliza
    sf.write(os.path.join(output_dir, "input_clean.wav"), y, sr)
    print("Audio original guardado como input_clean.wav")

    # === Añadir ruido ===
    snr_db = 5  # puedes cambiar a 0, 10, etc.
    y_noisy = add_noise(y, snr_db=snr_db)
    sf.write(os.path.join(output_dir, f"input_noisy_{snr_db}dB.wav"), y_noisy, sr)
    print(f"Añadido ruido (SNR={snr_db} dB) y guardado como input_noisy_{snr_db}dB.wav")

    # === Preparar tensores ===
    wav = torch.tensor(y_noisy).unsqueeze(0).unsqueeze(0)  # (1,1,T)

    # === Ejecutar ambos modos ===
    results = {}
    for mode in ["stft", "conv"]:
        res = run_mode(mode, output_dir, wav, sr, y_noisy, y)
        results[mode] = res

    # === Guardar resultados en archivo ===
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Ruido añadido: {snr_db} dB SNR\n\n")
        for mode, metrics in results.items():
            f.write(f"[{mode.upper()}]\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.3f}\n")
            f.write("\n")

    print("\nComparación completada.")
    print(f"Resultados guardados en: {output_dir}")
    print(results)


if __name__ == "__main__":
    demo()
