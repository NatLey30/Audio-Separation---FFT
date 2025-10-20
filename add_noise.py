import numpy as np


def add_noise(y, snr_db, noise=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if noise is None:
        noise = rng.standard_normal(len(y))
    # recorta/replica ruido a longitud
    if len(noise) < len(y):
        reps = int(np.ceil(len(y)/len(noise)))
        noise = np.tile(noise, reps)[:len(y)]
    else:
        noise = noise[:len(y)]
    Px = np.mean(y**2) + 1e-12
    Pn_target = Px / (10**(snr_db/10))
    Pn = np.mean(noise**2) + 1e-12
    noise = noise * np.sqrt(Pn_target/Pn)
    return (y + noise).astype(np.float32)
