"""Acoustic source models for aircraft (and other moving sources).

A source emits a pressure waveform s(t) referenced to 1 m (Pa). It is described by
  - a *shaft rate* f_shaft(t) (Hz) with slow random RPM wander, integrated to a phase;
  - a set of tonal components at fixed multiples of the shaft rate
    (propeller blade-passing harmonics, engine firing orders, rotor harmonics, fan tones);
  - a broadband component with a parametric spectrum.

Levels are dB re 20 uPa at 1 m. Presets are order-of-magnitude realistic and will be
calibrated against real recordings (see PLAN.md, simulator calibration loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

P_REF = 20e-6
F64 = NDArray[np.float64]


def db_to_pa(db: float | F64) -> float | F64:
    return P_REF * 10.0 ** (np.asarray(db) / 20.0) * np.sqrt(2.0)  # peak amplitude of a sinusoid


@dataclass(frozen=True)
class Tone:
    order: float  # multiple of the shaft rate
    level_db: float  # dB re 20 uPa at 1 m (RMS)
    directivity: float = 0.0  # 0 = omni; >0 favours the plane of rotation (sin^2 lobe)


@dataclass(frozen=True)
class Broadband:
    level_db: float  # overall dB re 20 uPa at 1 m
    peak_hz: float = 300.0  # spectral peak
    slope_lo_db_oct: float = 6.0  # rise below peak
    slope_hi_db_oct: float = -6.0  # fall above peak


@dataclass(frozen=True)
class SourceModel:
    name: str
    aircraft_class: str  # piston | turboprop | helicopter | jet | ground
    shaft_hz: float  # nominal shaft (or rotor) rate
    tones: tuple[Tone, ...] = ()
    broadband: Broadband | None = None
    rpm_wander: float = 0.003  # relative std of slow RPM drift
    rpm_tau_s: float = 20.0  # correlation time of RPM drift
    meta: dict = field(default_factory=dict)

    @property
    def fundamental_hz(self) -> float:
        """The strongest tone's nominal frequency - what a spectrogram reader would call f0."""
        if not self.tones:
            return 0.0
        t = max(self.tones, key=lambda x: x.level_db)
        return t.order * self.shaft_hz


def _prop(
    name: str,
    cls: str,
    rpm: float,
    blades: int,
    cylinders: int,
    bpf_db: float,
    bb_db: float,
    n_harm: int = 8,
    harm_decay_db: float = 4.0,
    firing_db: float = -8.0,
    bb_peak: float = 250.0,
    meta=None,
) -> SourceModel:
    shaft = rpm / 60.0
    tones = [Tone(blades * k, bpf_db - harm_decay_db * (k - 1), directivity=0.5) for k in range(1, n_harm + 1)]
    if cylinders:
        fire = cylinders / 2.0  # four-stroke firing order (per shaft revolution)
        for k in range(1, 5):
            order = fire * k
            if any(abs(order - t.order) < 1e-9 for t in tones):
                continue
            tones.append(Tone(order, bpf_db + firing_db - 5.0 * (k - 1)))
        tones.append(Tone(0.5 * fire, bpf_db + firing_db - 6.0))  # exhaust half-order
    return SourceModel(name, cls, shaft, tuple(tones), Broadband(bb_db, bb_peak), meta=meta or {})


# ---- presets ---------------------------------------------------------------
# BPF = blades * RPM / 60.  Levels: GA flyover ~70-75 dBA at 300 m  =>  ~120 dB at 1 m.

PRESETS: dict[str, SourceModel] = {
    "C172": _prop(
        "Cessna 172 (piston, 2-blade)", "piston", 2450, 2, 4, bpf_db=124, bb_db=118, meta={"icao_type": "C172"}
    ),
    "PA28": _prop(
        "Piper PA-28 (piston, 2-blade)", "piston", 2500, 2, 4, bpf_db=123, bb_db=117, meta={"icao_type": "P28A"}
    ),
    "SR22": _prop(
        "Cirrus SR22 (piston 6-cyl, 3-blade)", "piston", 2600, 3, 6, bpf_db=125, bb_db=119, meta={"icao_type": "SR22"}
    ),
    "BE20": _prop(
        "King Air 200 (turboprop, 4-blade)",
        "turboprop",
        1700,
        4,
        0,
        bpf_db=128,
        bb_db=122,
        n_harm=10,
        harm_decay_db=3.0,
        bb_peak=400,
        meta={"icao_type": "BE20"},
    ),
    "C208": _prop(
        "Cessna Caravan (turboprop, 3-blade)",
        "turboprop",
        1900,
        3,
        0,
        bpf_db=127,
        bb_db=121,
        n_harm=10,
        harm_decay_db=3.0,
        bb_peak=400,
        meta={"icao_type": "C208"},
    ),
    "DH8D": _prop(
        "Dash 8 Q400 (turboprop, 6-blade)",
        "turboprop",
        1020,
        6,
        0,
        bpf_db=130,
        bb_db=126,
        n_harm=8,
        harm_decay_db=3.5,
        bb_peak=500,
        meta={"icao_type": "DH8D"},
    ),
    "R44": SourceModel(
        "Robinson R44 (helicopter)",
        "helicopter",
        400 / 60.0,
        tones=tuple(
            [Tone(2 * k, 126 - 2.0 * (k - 1)) for k in range(1, 16)]  # main rotor BPF harmonics (blade slap)
            + [Tone(2 * 2426 / 400 * k, 118 - 4.0 * (k - 1)) for k in range(1, 6)]
        ),  # tail rotor
        broadband=Broadband(120, 400),
        meta={"icao_type": "R44"},
    ),
    "A320": SourceModel(
        "Airbus A320 (turbofan)",
        "jet",
        3300 / 60.0,  # N1 fan shaft rate at approach-ish power
        tones=(Tone(36, 118), Tone(72, 110)),  # fan BPF (36 blades) and 2nd harmonic
        broadband=Broadband(136, 400, 6.0, -5.0),
        meta={"icao_type": "A320"},
    ),
}


# ---- RPM wander --------------------------------------------------------------


def ou_process(n: int, dt: float, tau: float, sigma: float, rng: np.random.Generator) -> F64:
    """Ornstein-Uhlenbeck process with stationary std `sigma` and correlation time `tau`."""
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1.0 - a * a)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    eps = rng.normal(0.0, 1.0, n)
    for i in range(1, n):  # n is small (source control grid, ~10 Hz)
        x[i] = a * x[i - 1] + b * eps[i]
    return x


@dataclass(frozen=True)
class ShaftPhase:
    """Shaft phase phi(t) (radians, cumulative) on a control grid, for cubic interpolation."""

    t: F64
    phi: F64
    f_shaft: F64

    def __call__(self, tq: F64) -> F64:
        # Cubic Hermite with derivative 2*pi*f_shaft: exact for piecewise-smooth rates.
        t, phi, d = self.t, self.phi, 2 * np.pi * self.f_shaft
        i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, t.size - 2)
        h = t[i + 1] - t[i]
        s = (tq - t[i]) / h
        s2, s3 = s * s, s * s * s
        return (
            (2 * s3 - 3 * s2 + 1) * phi[i]
            + (s3 - 2 * s2 + s) * h * d[i]
            + (-2 * s3 + 3 * s2) * phi[i + 1]
            + (s3 - s2) * h * d[i + 1]
        )


def make_shaft_phase(
    src: SourceModel, t0: float, t1: float, rng: np.random.Generator, dt: float = 0.05, rpm_profile: F64 | None = None
) -> ShaftPhase:
    t = np.arange(t0 - 1.0, t1 + 1.0 + dt, dt)
    rel = ou_process(t.size, dt, src.rpm_tau_s, src.rpm_wander, rng)
    f = src.shaft_hz * (1.0 + rel) * (1.0 if rpm_profile is None else rpm_profile)
    phi0 = rng.uniform(0, 2 * np.pi)
    phi = phi0 + 2 * np.pi * np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
    return ShaftPhase(t, phi, f)
