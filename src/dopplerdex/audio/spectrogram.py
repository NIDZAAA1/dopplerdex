"""Spectrogram computation, noise whitening and log-frequency resampling."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT, resample_poly
from scipy.signal.windows import get_window

F64 = NDArray[np.float64]


def load_audio(path: str | Path, fs: int = 16_000, mono: bool = True) -> tuple[F64, int]:
    """Decode any audio/video file with ffmpeg to float32 at `fs` (mono by default)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ar", str(fs)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), fs


def resample(x: F64, fs_in: int, fs_out: int) -> F64:
    if fs_in == fs_out:
        return x
    g = np.gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


@dataclass(frozen=True)
class Spectrogram:
    t: F64  # frame centre times (s), relative to the start of x (+ t0)
    f: F64  # bin frequencies (Hz)
    power: F64  # (F, T) power spectral density (linear)
    fs: float
    nperseg: int
    hop: int

    def db(self) -> F64:
        return 10.0 * np.log10(self.power + 1e-30)

    def band(self, f_lo: float, f_hi: float) -> Spectrogram:
        m = (self.f >= f_lo) & (self.f <= f_hi)
        return Spectrogram(self.t, self.f[m], self.power[m], self.fs, self.nperseg, self.hop)

    def time_slice(self, t_lo: float, t_hi: float) -> Spectrogram:
        m = (self.t >= t_lo) & (self.t <= t_hi)
        return Spectrogram(self.t[m], self.f, self.power[:, m], self.fs, self.nperseg, self.hop)


def spectrogram(
    x: F64,
    fs: float,
    nperseg: int = 4096,
    hop: int | None = None,
    window: str = "hann",
    t0: float = 0.0,
    zero_pad: int = 1,
) -> Spectrogram:
    hop = hop or nperseg // 4
    win = get_window(window, nperseg, fftbins=True)
    stft = ShortTimeFFT(win, hop=hop, fs=fs, mfft=nperseg * zero_pad, scale_to="psd")
    P = stft.spectrogram(x)
    t = stft.t(x.size) + t0
    keep = (t >= t0) & (t <= t0 + x.size / fs)
    return Spectrogram(t[keep], stft.f, P[:, keep], fs, nperseg, hop)


def whiten(spec: Spectrogram, percentile: float = 50.0, smooth_bins: int = 0) -> Spectrogram:
    """Divide each frequency row by its noise floor (a percentile over time).

    Returns an SNR-like spectrogram (1.0 = noise floor). This removes the spectral tilt of the
    background and makes faint tonal tracks comparable across frequency.
    """
    floor = np.percentile(spec.power, percentile, axis=1, keepdims=True)
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        floor = np.convolve(floor[:, 0], k, mode="same")[:, None]
    return Spectrogram(spec.t, spec.f, spec.power / (floor + 1e-30), spec.fs, spec.nperseg, spec.hop)


def to_log_frequency(spec: Spectrogram, f_lo: float, f_hi: float, bins_per_octave: int = 96) -> Spectrogram:
    """Resample the frequency axis to log spacing (Doppler and f0 become pure shifts)."""
    n = int(np.ceil(np.log2(f_hi / f_lo) * bins_per_octave)) + 1
    fl = f_lo * 2.0 ** (np.arange(n) / bins_per_octave)
    P = np.empty((n, spec.power.shape[1]))
    for j in range(spec.power.shape[1]):
        P[:, j] = np.interp(fl, spec.f, spec.power[:, j])
    return Spectrogram(spec.t, fl, P, spec.fs, spec.nperseg, spec.hop)
