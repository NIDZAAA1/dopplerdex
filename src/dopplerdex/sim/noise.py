"""Ambient noise environment for synthetic recordings.

Components (all in Pa at the microphone):
  - background: stationary coloured noise with a configurable spectral tilt
  - wind: low-frequency rumble with gusty (OU-modulated) amplitude
  - mains hum: 50/60 Hz + harmonics with a random-walk ENF (as in real grid recordings)
  - steady tones: HVAC / machinery lines at fixed frequencies (seen in real spectrograms)
  - birds: short FM chirps (1.5-7 kHz)
  - transients: broadband clicks/bumps (doors, footsteps)

`NoiseProfile.fit_band_levels` + `texture_noise` reproduce the *statistics* of a real
recording (per-band level distribution over time) for realistic synthetic backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from dopplerdex.sim.sources import db_to_pa, ou_process

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class AmbientConfig:
    background_db: float = 42.0  # overall SPL of the stationary background
    background_tilt_db_oct: float = -4.0  # spectral slope (urban ~ -3..-5)
    wind_db: float = 45.0  # mean level of low-frequency wind rumble (0 disables -> use None)
    wind_gustiness: float = 6.0  # dB std of gust modulation
    mains_hz: float | None = 60.0  # 50 or 60; None disables
    mains_db: float = 30.0
    enf_wander_hz: float = 0.02
    steady_tones: tuple[tuple[float, float], ...] = ((118.0, 28.0), (241.0, 24.0))  # (Hz, dB)
    birds_per_min: float = 2.0
    bird_db: float = 50.0
    transients_per_min: float = 1.0
    transient_db: float = 60.0


def colored_noise(
    n: int,
    fs: float,
    level_db: float,
    tilt_db_oct: float,
    rng: np.random.Generator,
    f_ref: float = 500.0,
    f_lo: float = 10.0,
) -> F64:
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs)
    g = np.where(f > f_lo, (np.maximum(f, f_lo) / f_ref) ** (tilt_db_oct / (20 * np.log10(2))), 0.0)
    x = np.fft.irfft(spec * g, n)
    x /= np.sqrt(np.mean(x * x)) + 1e-30
    return x * db_to_pa(level_db) / np.sqrt(2)


def _lowpass_noise(n: int, fs: float, fc: float, rng: np.random.Generator) -> F64:
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs)
    spec *= 1.0 / np.sqrt(1.0 + (f / fc) ** 4)
    spec[f < 3.0] = 0.0
    x = np.fft.irfft(spec, n)
    return x / (np.sqrt(np.mean(x * x)) + 1e-30)


def render_ambient(cfg: AmbientConfig, fs: float, duration: float, rng: np.random.Generator) -> dict[str, F64]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    parts: dict[str, F64] = {}
    parts["background"] = colored_noise(n, fs, cfg.background_db, cfg.background_tilt_db_oct, rng)

    if cfg.wind_db is not None and cfg.wind_db > 0:
        ctrl_dt = 0.1
        tc = np.arange(0, duration + ctrl_dt, ctrl_dt)
        gust_db = ou_process(tc.size, ctrl_dt, 4.0, cfg.wind_gustiness, rng)
        env = 10 ** (np.interp(t, tc, gust_db) / 20.0)
        parts["wind"] = _lowpass_noise(n, fs, 80.0, rng) * env * db_to_pa(cfg.wind_db) / np.sqrt(2)

    if cfg.mains_hz:
        ctrl_dt = 1.0
        tc = np.arange(0, duration + 2, ctrl_dt)
        enf = cfg.mains_hz + np.clip(np.cumsum(rng.normal(0, cfg.enf_wander_hz / 4, tc.size)), -0.05, 0.05)
        phi = 2 * np.pi * np.cumsum(np.interp(t, tc, enf)) / fs
        hum = sum(db_to_pa(cfg.mains_db - 6 * (k - 1)) * np.sin(k * phi + rng.uniform(0, 6.3)) for k in (1, 2, 3, 4))
        parts["mains"] = np.asarray(hum)
        parts["_enf_hz"] = np.interp(np.arange(0, duration, 1.0), tc, enf)

    if cfg.steady_tones:
        parts["tones"] = sum(
            db_to_pa(db) * np.sin(2 * np.pi * (f0 * (1 + 1e-4 * rng.normal())) * t + rng.uniform(0, 6.3))
            for f0, db in cfg.steady_tones
        )

    ev = np.zeros(n)
    for _ in range(rng.poisson(cfg.birds_per_min * duration / 60.0)):
        dur = rng.uniform(0.08, 0.4)
        m = int(dur * fs)
        start = rng.integers(0, max(n - m, 1))
        tt = np.arange(m) / fs
        f_a, f_b = rng.uniform(1500, 7000, 2)
        ph = 2 * np.pi * (f_a * tt + 0.5 * (f_b - f_a) / dur * tt**2)
        chirp = np.sin(ph) * np.sin(np.pi * tt / dur) ** 2
        ev[start : start + m] += db_to_pa(cfg.bird_db + rng.normal(0, 5)) * chirp[: n - start]
    for _ in range(rng.poisson(cfg.transients_per_min * duration / 60.0)):
        m = int(rng.uniform(0.01, 0.15) * fs)
        start = rng.integers(0, max(n - m, 1))
        burst = rng.normal(size=m) * np.exp(-np.arange(m) / (0.2 * m))
        ev[start : start + m] += db_to_pa(cfg.transient_db + rng.normal(0, 6)) * burst[: n - start] / np.sqrt(2)
    parts["events"] = ev
    return parts


# ---- statistics matching --------------------------------------------------------


@dataclass(frozen=True)
class NoiseProfile:
    """Per-band level time series of a real recording (dB, arbitrary reference)."""

    band_edges: F64  # (B+1,) Hz
    frame_dt: float
    levels_db: F64  # (B, T)

    def scaled(self, gain_db: float) -> NoiseProfile:
        """Shift all band levels (e.g. convert digital-units statistics to pascals)."""
        return NoiseProfile(self.band_edges, self.frame_dt, self.levels_db + gain_db)

    @staticmethod
    def fit_band_levels(
        x: F64, fs: float, bands_per_oct: int = 3, f_lo: float = 20.0, frame_s: float = 0.5
    ) -> NoiseProfile:
        f_hi = fs / 2 * 0.9
        n_b = int(np.floor(np.log2(f_hi / f_lo) * bands_per_oct))
        edges = f_lo * 2 ** (np.arange(n_b + 1) / bands_per_oct)
        nper = int(2 ** np.ceil(np.log2(frame_s * fs)))
        stft = ShortTimeFFT(hann(nper, sym=False), hop=nper // 2, fs=fs, scale_to="psd")
        P = stft.spectrogram(x)
        f = stft.f
        lv = np.stack(
            [
                10 * np.log10(P[(f >= a) & (f < b)].mean(axis=0) + 1e-20)
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
        return NoiseProfile(edges, stft.hop / fs, lv)


def texture_noise(
    profile: NoiseProfile, fs: float, duration: float, rng: np.random.Generator, block_s: float = 30.0
) -> F64:
    """Synthesise noise whose per-band level *distribution and temporal correlation* match the profile.

    Band level trajectories are resampled as contiguous blocks from the profile (preserves
    non-stationarity statistics such as gusts and traffic), then imposed on random-phase noise
    in the STFT domain.
    """
    n = int(round(duration * fs))
    nper = 2048
    stft = ShortTimeFFT(hann(nper, sym=False), hop=nper // 2, fs=fs, scale_to="psd")
    n_frames = stft.p_max(n) - stft.p_min
    T = profile.levels_db.shape[1]
    blk = max(1, int(block_s / profile.frame_dt))
    lv = []
    while sum(b.shape[1] for b in lv) < int(duration / profile.frame_dt) + 2:
        s = rng.integers(0, max(T - blk, 1))
        lv.append(profile.levels_db[:, s : s + blk])
    lv = np.concatenate(lv, axis=1)
    t_prof = np.arange(lv.shape[1]) * profile.frame_dt
    tf = stft.t(n)
    f = stft.f
    centers = np.sqrt(profile.band_edges[:-1] * profile.band_edges[1:])
    lv_t = np.stack([np.interp(tf, t_prof, row) for row in lv])  # (B, frames)
    # interpolate band levels across frequency (log f)
    logf = np.log(np.maximum(f, centers[0]))
    lv_f = np.stack([np.interp(logf, np.log(centers), lv_t[:, k]) for k in range(lv_t.shape[1])], axis=1)
    amp = 10 ** (lv_f / 20.0)
    ph = rng.uniform(0, 2 * np.pi, amp.shape)
    S = amp * np.exp(1j * ph)
    S *= f[:, None] >= profile.band_edges[0]
    x = stft.istft(S[:, :n_frames], k1=n)
    # Calibrate the absolute level: match the mean band level of the profile.
    out_prof = NoiseProfile.fit_band_levels(x, fs, f_lo=profile.band_edges[0], frame_s=profile.frame_dt * 2)
    k = min(out_prof.levels_db.shape[0], profile.levels_db.shape[0])
    gain_db = np.median(profile.levels_db[:k].mean(axis=1) - out_prof.levels_db[:k].mean(axis=1))
    return x * 10 ** (gain_db / 20.0)
