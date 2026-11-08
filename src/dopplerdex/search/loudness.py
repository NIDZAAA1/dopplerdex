"""Predicted vs observed loudness envelopes.

Used for (a) aligning a recording's clock to UTC when the location is known (oracle / known-location
mode), and (b) later the audibility/silence term of the scorer: a candidate location predicts when
aircraft should have been loud; the recording must agree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dopplerdex.audio.spectrogram import spectrogram
from dopplerdex.physics.atmosphere import absorption_db_per_m
from dopplerdex.search.oracle import PredictedPass

F64 = NDArray[np.float64]

# Broadband source level (dB re 20 uPa @ 1 m, 60-1500 Hz band) by class - order of magnitude, to calibrate.
SOURCE_DB = {"jet": 142.0, "turboprop": 132.0, "piston": 126.0, "helicopter": 132.0, "unknown": 128.0, "other": 120.0}


def predicted_level(passes: list[PredictedPass], t_grid: F64, f_ref: float = 400.0, ambient_db: float = 45.0) -> F64:
    """Sum of predicted band levels (dB) at the mic on t_grid (unix s), on top of an ambient floor
    (without the floor, aircraft-free stretches produce fake -inf dips that dominate any comparison)."""
    alpha = float(absorption_db_per_m(f_ref, 15.0, 70.0))
    p2 = np.zeros_like(t_grid)
    for p in passes:
        r = np.interp(t_grid, p.t_r, np.where(p.valid, p.rng, 1e9), left=1e9, right=1e9)
        lv = SOURCE_DB.get(p.flight.aircraft_class, 128.0) - 20 * np.log10(np.maximum(r, 30.0)) - alpha * r
        p2 += 10 ** (lv / 10)
    return 10 * np.log10(p2 + 10 ** (ambient_db / 10))


def observed_level(
    x: F64, fs: float, f_lo: float = 300.0, f_hi: float = 3000.0, frame_s: float = 1.0
) -> tuple[F64, F64]:
    """Band level (dB, arbitrary reference) per frame_s of the recording; returns (t, level)."""
    n = int(2 ** np.ceil(np.log2(frame_s * fs)))
    sp = spectrogram(x, fs, nperseg=n, hop=int(frame_s * fs)).band(f_lo, f_hi)
    return sp.t, 10 * np.log10(sp.power.sum(axis=0) + 1e-30)


def _excess(y: F64, win: int, pct: float = 10.0) -> F64:
    """dB above a running low-percentile baseline (busy airspace keeps the median high, so use a low percentile)."""
    from scipy.ndimage import percentile_filter

    return np.clip(y - percentile_filter(y, pct, size=win, mode="nearest"), 0, None)


def align_by_loudness(
    t_rec: F64, level_rec: F64, passes: list[PredictedPass], t_lo: float, t_hi: float, step: float = 1.0
) -> tuple[float, F64, F64]:
    """Find the recording start T0 (unix s) maximising the correlation between observed and predicted
    loudness excess. Returns (T0, candidate T0 grid, correlation)."""
    dur = float(t_rec[-1] - t_rec[0])
    grid = np.arange(t_lo, t_hi + dur + 1, step)
    win = int(600 / step) | 1
    pred = _excess(predicted_level(passes, grid), win)  # only "louder than usual" carries timing information
    obs = _excess(np.interp(np.arange(0, dur, step), t_rec - t_rec[0], level_rec), win)
    obs = (obs - obs.mean()) / (obs.std() + 1e-9)
    m = obs.size
    # sliding normalised correlation via FFT
    n = 1 << int(np.ceil(np.log2(pred.size + m)))
    num = np.fft.irfft(np.fft.rfft(pred, n) * np.conj(np.fft.rfft(obs, n)), n)[: pred.size - m + 1]
    c1 = np.concatenate([[0.0], np.cumsum(pred)])
    c2 = np.concatenate([[0.0], np.cumsum(pred**2)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    std = np.sqrt(np.maximum(s2 / m - (s1 / m) ** 2, 1e-12))
    corr = num / (m * std)
    k = int(np.argmax(corr))
    t0s = grid[: corr.size] - (t_rec[0])
    return float(t0s[k]), t0s, corr
