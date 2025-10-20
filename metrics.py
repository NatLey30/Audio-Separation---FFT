import torch, numpy as np

def si_sdr(est, ref, eps=1e-8):
    # est/ref: (T,) torch
    s = ref - ref.mean()
    sh = est - est.mean()
    s_target = (torch.dot(sh, s) / (torch.dot(s, s)+eps)) * s
    e_noise = sh - s_target
    return 10 * torch.log10((torch.sum(s_target**2)+eps)/(torch.sum(e_noise**2)+eps))

def batch_si_sdr(ests, refs):
    # ests/refs: (B,S,T)
    B,S,T = ests.shape
    out = []
    for b in range(B):
        vals = []
        for s in range(S):
            vals.append(si_sdr(ests[b,s], refs[b,s]).item())
        out.append(vals)
    return np.array(out)  # (B,S)

def lsd_db(ref_mag, est_mag, eps=1e-8):
    # ref_mag/est_mag: (F,Frames) magnitudes (numpy)
    return np.sqrt(np.mean((20*np.log10(est_mag+eps)-20*np.log10(ref_mag+eps))**2))

def snri_db(mixture, est, target):
    # mejora de SNR respecto a la mezcla
    def snr(x, n, eps=1e-8):
        return 10*np.log10((np.sum(x**2)+eps)/(np.sum(n**2)+eps))
    mix_noise = mixture - target
    est_noise = est - target
    return snr(target, est_noise) - snr(target, mix_noise)
