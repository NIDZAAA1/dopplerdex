"""Time-frequency features for tonal-track detection.

Pipeline:  STFT power  ->  2-D whitening  ->  log-frequency  ->  harmonic sum  ->  robust z-score.

Why log-frequency: a Doppler shift multiplies every harmonic by the same factor D(t), so on a
log-f axis the whole harmonic comb moves rigidly, and harmonic k sits exactly log2(k) octaves
above the fundamental. Summing those rows (harmonic sum) integrates the energy of all harmonics
into one ridge, which raises faint curves above the noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dopplerdex.audio.spectrogram import Spectrogram, spectrogram

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class LogFreqMap:
    t: F64  # frame times (s)
    f: F64  # log-spaced bin centre frequencies (Hz)
    bpo: int  # bins per octave
    z: F32  # (F, T) feature map

    def bin_of(self, freq_hz: F64 | float) -> F64:
        return np.log2(np.asarray(freq_hz) / self.f[0]) * self.bpo

    def freq_of(self, bins: F64 | float) -> F64:
        return self.f[0] * 2.0 ** (np.asarray(bins, dtype=np.float64) / self.bpo)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def whiten_2d(power: F64, row_pct: float = 50.0, col_pct: float = 50.0) -> F32:
    """Divide by the per-frequency noise floor (over time), then by the per-frame floor (over frequency).

    Row whitening removes the background's spectral tilt and steady tones' baseline; column
    whitening suppresses broadband bursts (wind gusts, traffic, door slams).
    """
    p = power.astype(np.float32, copy=False)
    p = p / (np.percentile(p, row_pct, axis=1, keepdims=True) + 1e-30)
    p = p / (np.percentile(p, col_pct, axis=0, keepdims=True) + 1e-30)
    return p


def to_logf(spec: Spectrogram, data: F32, f_lo: float, f_hi: float, bpo: int) -> tuple[F64, F32]:
    """Linear-interpolate rows of `data` (aligned with spec.f) onto a log-frequency grid."""
    n = int(np.floor(np.log2(f_hi / f_lo) * bpo)) + 1
    fl = f_lo * 2.0 ** (np.arange(n) / bpo)
    df = spec.f[1] - spec.f[0]
    x = (fl - spec.f[0]) / df
    i0 = np.clip(np.floor(x).astype(int), 0, spec.f.size - 2)
    w = (x - i0).astype(np.float32)[:, None]
    return fl, data[i0] * (1 - w) + data[i0 + 1] * w


def running_floor_whiten(
    z: F32, frame_dt: float, window_s: float = 60.0, block_s: float = 6.0, pct: float = 50.0
) -> F32:
    """Divide each frequency row by a *running* noise floor over ~window_s.

    Steady or slowly-switching tones (machinery, HVAC, hum) dominate their row for minutes and
    are normalised away; a Doppler line sweeps through a given bin in seconds and survives.
    Implemented as block percentiles (block_s) + running median over blocks, interpolated back.
    """
    n_f, n_t = z.shape
    bl = max(1, int(round(block_s / frame_dt)))
    nb = int(np.ceil(n_t / bl))
    pad = nb * bl - n_t
    zp = np.pad(z, ((0, 0), (0, pad)), mode="edge").reshape(n_f, nb, bl)
    blocks = np.percentile(zp, pct, axis=2)  # (F, nb)
    k = max(1, int(round(window_s / block_s)) | 1)
    from scipy.ndimage import median_filter

    floor_b = median_filter(blocks, size=(1, k), mode="nearest")
    centres = (np.arange(nb) + 0.5) * bl
    t = np.arange(n_t)
    floor = np.stack([np.interp(t, centres, row) for row in floor_b]).astype(np.float32)
    return (z / (floor + 1e-6)).astype(np.float32)


def tonal_contrast(z: F32, half_width_bins: int) -> F32:
    """Divide by a running median along frequency: narrow lines stay, broad structure (engine roar,
    ground-reflection fringes, wind) is flattened to ~1."""
    from scipy.ndimage import median_filter

    return (z / (median_filter(z, size=(2 * half_width_bins + 1, 1), mode="nearest") + 1e-6)).astype(np.float32)


def harmonic_sum(z: F32, bpo: int, n_harm: int = 6, clip: float = 25.0) -> F32:
    """H(i) = mean_k min(z(i + bpo*log2 k), clip) over k = 1..n_harm (bins beyond the top are ignored)."""
    zc = np.minimum(z, clip)
    out = np.zeros_like(zc)
    cnt = np.zeros((z.shape[0], 1), dtype=np.float32)
    for k in range(1, n_harm + 1):
        o = int(round(np.log2(k) * bpo))
        if o >= z.shape[0]:
            break
        out[: z.shape[0] - o] += zc[o:]
        cnt[: z.shape[0] - o] += 1
    return out / np.maximum(cnt, 1)


def robust_z(h: F32) -> F32:
    """Per-row robust z-score: (h - median) / (1.4826 * MAD), so thresholds mean the same at every f."""
    med = np.median(h, axis=1, keepdims=True)
    mad = np.median(np.abs(h - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    return ((h - med) / mad).astype(np.float32)


@dataclass(frozen=True)
class FeatureConfig:
    nperseg: int = 4096  # at 8 kHz: 0.51 s window, 1.95 Hz bins
    hop: int = 1024  # 0.128 s
    f_lo: float = 35.0
    f_hi: float = 1600.0
    bpo: int = 96
    n_harm: int = 6
    contrast_half_width: int = 10  # bins (~1/10 octave at 96 bpo); 0 disables the tonality filter
    mode: str = "single"  # "single": track individual lines; "harmonic": harmonic-sum map; "max": either
    time_smooth_frames: int = 3  # light smoothing along time raises faint-line SNR
    steady_window_s: float = 60.0  # running-floor whitening window; 0 disables


def tonal_feature_map(
    x: F64, fs: float, cfg: FeatureConfig = FeatureConfig(), t0: float = 0.0
) -> tuple[LogFreqMap, LogFreqMap]:
    """Return (whitened log-f map, harmonic-sum z-map)."""
    spec = spectrogram(x, fs, nperseg=cfg.nperseg, hop=cfg.hop, t0=t0)
    spec = spec.band(max(cfg.f_lo * 0.9, 0.0), min(cfg.f_hi * 1.05, fs / 2))
    w = whiten_2d(spec.power)
    fl, zl = to_logf(spec, w, cfg.f_lo, cfg.f_hi, cfg.bpo)
    if cfg.steady_window_s > 0:
        zl = running_floor_whiten(zl, float(spec.t[1] - spec.t[0]), cfg.steady_window_s)
    if cfg.contrast_half_width > 0:
        zl = tonal_contrast(zl, cfg.contrast_half_width)
    if cfg.time_smooth_frames > 1:
        from scipy.ndimage import uniform_filter1d

        zl = uniform_filter1d(zl, cfg.time_smooth_frames, axis=1).astype(np.float32)
    if cfg.mode == "single":
        feat = robust_z(np.log1p(zl))
    elif cfg.mode == "harmonic":
        feat = robust_z(harmonic_sum(zl, cfg.bpo, cfg.n_harm))
    else:
        feat = np.maximum(robust_z(np.log1p(zl)), robust_z(harmonic_sum(zl, cfg.bpo, cfg.n_harm)))
    return LogFreqMap(spec.t, fl, cfg.bpo, zl), LogFreqMap(spec.t, fl, cfg.bpo, feat)
