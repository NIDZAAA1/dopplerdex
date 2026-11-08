"""Physically-based rendering of moving acoustic sources at a stationary microphone.

The microphone pressure is a sum over propagation paths (direct + ground-reflected image):

    p(t_r) = sum_paths  A_path(t_e) / r_path(t_e) * s(t_e(t_r))

where t_e(t_r) is the retarded (emission) time for that path. Evaluating the source
waveform at the emission time produces the Doppler shift, the propagation delay and the
Lloyd's-mirror interference *automatically* - nothing is "added" as an effect. Atmospheric
absorption (frequency and range dependent) is applied as a slowly time-varying filter in the
STFT domain; turbulence adds slow log-normal amplitude scintillation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import ShortTimeFFT, sosfilt
from scipy.signal.windows import hann

from dopplerdex.physics.atmosphere import absorption_db_per_m, speed_of_sound
from dopplerdex.physics.doppler import RetardedSolution, solve_retarded
from dopplerdex.physics.propagation import image_observer, plane_wave_reflection
from dopplerdex.physics.trajectory import Trajectory
from dopplerdex.sim.sources import SourceModel, db_to_pa, make_shaft_phase, ou_process

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Atmosphere:
    temp_c: float = 15.0
    rel_humidity: float = 60.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)  # ENU m/s (air velocity)
    turbulence_db: float = 1.5  # std of slow amplitude scintillation (dB)
    turbulence_tau_s: float = 1.5

    @property
    def c(self) -> float:
        return float(speed_of_sound(self.temp_c, self.rel_humidity))


@dataclass(frozen=True)
class Ground:
    mic_height: float = 1.5  # m above local ground (ground plane z = 0)
    flow_resistivity: float | None = 300e3  # Pa s m^-2 (grass ~ 2-3e5); None = rigid
    enabled: bool = True


@dataclass
class RenderedSource:
    signal: F64  # pressure at the mic (Pa), full output length
    direct: RetardedSolution  # on the control grid
    control_t: F64
    shaft_at_emission: F64  # shaft rate f_shaft(t_e) on the control grid (Hz)
    tone_orders: F64
    source: SourceModel
    active: NDArray[np.bool_]
    extras: dict = field(default_factory=dict)

    def received_tone_freqs(self) -> F64:
        """(n_tones, n_control) instantaneous received frequency of each tone - ground truth curves."""
        return self.tone_orders[:, None] * (self.shaft_at_emission * self.direct.doppler)[None, :]


def _shaped_noise(n: int, fs: float, bb, rng: np.random.Generator) -> F64:
    """Gaussian noise with a two-slope spectrum peaking at bb.peak_hz, scaled to bb.level_db RMS."""
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    octs = np.log2(f / bb.peak_hz)
    gain_db = np.where(octs < 0, bb.slope_lo_db_oct * octs, bb.slope_hi_db_oct * octs)
    gain_db = np.where(f < 15.0, gain_db - 40.0, gain_db)  # no meaningful content below ~15 Hz
    x = np.fft.irfft(spec * 10 ** (gain_db / 20.0), n)
    x /= np.sqrt(np.mean(x * x)) + 1e-30
    return x * db_to_pa(bb.level_db) / np.sqrt(2.0)


def apply_absorption(x: F64, fs: float, t: F64, rng_m: F64, atm: Atmosphere, nperseg: int = 2048,
                     t_start: float = 0.0) -> F64:  # fmt: skip
    """Time-varying atmospheric absorption: per STFT frame, attenuate by alpha(f) * r(t).

    `t`/`rng_m` give the range on the absolute clock; `t_start` is the absolute time of x[0].
    """
    if x.size < nperseg:
        return x
    win = hann(nperseg, sym=False)
    stft = ShortTimeFFT(win, hop=nperseg // 4, fs=fs, scale_to=None)
    S = stft.stft(x)
    f = stft.f
    tf = stft.t(x.size) + t_start  # frame times on the same absolute clock as `t`
    r = np.interp(tf, t, rng_m)
    alpha = absorption_db_per_m(f, atm.temp_c, atm.rel_humidity)
    S *= 10.0 ** (-(alpha[:, None] * r[None, :]) / 20.0)
    return stft.istft(S, k1=x.size)


def render_source(
    traj: Trajectory,
    src: SourceModel,
    fs: float,
    t0: float,
    duration: float,
    atm: Atmosphere,
    ground: Ground,
    rng: np.random.Generator,
    ctrl_dt: float = 0.02,
    max_range: float = 25_000.0,
    absorption: bool = True,
) -> RenderedSource:
    """Render one moving source at the microphone located at (0, 0, ground.mic_height)."""
    c = atm.c
    w = np.asarray(atm.wind, dtype=np.float64)
    n_out = int(round(duration * fs))
    out = np.zeros(n_out)
    tc = t0 + np.arange(0.0, duration + ctrl_dt, ctrl_dt)
    mic = np.array([0.0, 0.0, ground.mic_height])
    direct = solve_retarded(traj, mic, tc, c, w)
    active = direct.valid & (direct.rng < max_range)
    orders = np.array([tn.order for tn in src.tones], dtype=np.float64)
    shaft = make_shaft_phase(src, float(direct.t_e.min()), float(direct.t_e.max()), rng)
    shaft_at_e = np.interp(direct.t_e, shaft.t, shaft.f_shaft)
    res = RenderedSource(out, direct, tc, shaft_at_e, orders, src, active)
    if not np.any(active):
        return res

    # Render only the active window (plus margin), on the full-rate grid.
    ia = np.flatnonzero(active)
    i0 = max(int((tc[ia[0]] - t0 - 1.0) * fs), 0)
    i1 = min(int((tc[ia[-1]] - t0 + 1.0) * fs), n_out)
    tr_full = t0 + np.arange(i0, i1) / fs
    # Smooth fade at the window edges (range gate) to avoid clicks.
    gate_c = np.clip((max_range - direct.rng) / (0.1 * max_range), 0.0, 1.0) * direct.valid

    paths: list[tuple[RetardedSolution, bool]] = [(direct, False)]
    if ground.enabled:
        paths.append((solve_retarded(traj, image_observer(mic, 0.0), tc, c, w), True))

    seg = np.zeros(i1 - i0)
    noise_src = None
    if src.broadband is not None:
        te_lo = min(p.t_e.min() for p, _ in paths) - 1.0
        te_hi = max(p.t_e.max() for p, _ in paths) + 1.0
        n_src = int((te_hi - te_lo) * fs) + 2
        noise_src = (te_lo, _shaped_noise(n_src, fs, src.broadband, rng))

    for sol, is_reflection in paths:
        te = CubicSpline(tc, sol.t_e)(tr_full)
        inv_r = np.interp(tr_full, tc, gate_c / sol.rng)
        vhat = sol.src_vel / (np.linalg.norm(sol.src_vel, axis=1, keepdims=True) + 1e-9)
        cos_th = np.einsum("ij,ij->i", vhat, sol.n)
        sin2 = np.interp(tr_full, tc, 1.0 - cos_th**2)
        if is_reflection:
            grazing = np.arcsin(np.clip(sol.src_pos[:, 2] / sol.rng, 0.0, 1.0))
        phi = shaft(te)
        for tone, order in zip(src.tones, orders, strict=True):
            amp = db_to_pa(tone.level_db) * inv_r * (1.0 - tone.directivity + tone.directivity * sin2)
            ph = 0.0
            if is_reflection:
                f_rx = order * shaft_at_e * sol.doppler
                R = plane_wave_reflection(f_rx, grazing, ground.flow_resistivity)
                amp = amp * np.interp(tr_full, tc, np.abs(R))
                ph = np.interp(tr_full, tc, np.unwrap(np.angle(R)))
            seg += amp * np.sin(order * phi + ph)
        if noise_src is not None:
            te_lo, nz = noise_src
            g = 1.0
            if is_reflection:
                g = float(
                    np.median(np.real(plane_wave_reflection(np.array([300.0]), grazing, ground.flow_resistivity)))
                )
            seg += g * inv_r * np.interp((te - te_lo) * fs, np.arange(nz.size), nz)

    if atm.turbulence_db > 0:
        sc = ou_process(tc.size, ctrl_dt, atm.turbulence_tau_s, atm.turbulence_db, rng)
        seg *= 10 ** (np.interp(tr_full, tc, sc) / 20.0)
    if absorption:
        seg = apply_absorption(seg, fs, tc, direct.rng, atm, t_start=float(tr_full[0]))
    out[i0:i1] = seg
    res.signal = out
    return res


# ---- device chain --------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """Microphone + recorder. full_scale_db: SPL (dB re 20 uPa, RMS of a full-scale sine)."""

    full_scale_db: float = 110.0
    highpass_hz: float | None = 40.0
    bits: int = 16
    self_noise_db: float = 25.0  # equivalent input noise (broadband, dB SPL)


def apply_device(p: F64, fs: float, dev: Device, rng: np.random.Generator) -> F64:
    from scipy.signal import butter

    x = p + rng.normal(0.0, 20e-6 * 10 ** (dev.self_noise_db / 20.0), p.size)
    if dev.highpass_hz:
        x = sosfilt(butter(2, dev.highpass_hz, "highpass", fs=fs, output="sos"), x)
    fs_pa = db_to_pa(dev.full_scale_db)
    y = np.clip(x / fs_pa, -1.0, 1.0)
    q = 2.0 ** (dev.bits - 1)
    return np.round(y * q) / q
