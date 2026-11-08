"""Scenario description -> rendered synthetic recording + ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dopplerdex.physics.trajectory import Trajectory
from dopplerdex.sim.noise import AmbientConfig, NoiseProfile, render_ambient, texture_noise
from dopplerdex.sim.render import Atmosphere, Device, Ground, RenderedSource, apply_device, render_source
from dopplerdex.sim.sources import PRESETS, SourceModel

F64 = NDArray[np.float64]


@dataclass
class AircraftPass:
    traj: Trajectory
    source: SourceModel
    label: str = ""


@dataclass(frozen=True)
class EventTruth:
    label: str
    aircraft_class: str
    t_cpa_source: float  # when the aircraft is at closest approach
    t_cpa_heard: float  # when the CPA emission reaches the mic
    r_cpa: float  # slant range at CPA (m)
    speed_cpa: float  # ground speed at CPA (m/s)
    alt_cpa: float  # height above mic ground (m)
    f0: float  # nominal strongest-tone frequency (Hz)
    control_t: F64 = field(repr=False)
    tone_freqs: F64 = field(repr=False)  # (n_tones, n_control) received frequencies
    tone_levels_db: F64 = field(repr=False)  # nominal source level per tone (dB @ 1 m)
    active: NDArray[np.bool_] = field(repr=False)
    range_m: F64 = field(repr=False)


@dataclass
class Scenario:
    aircraft: list[AircraftPass]
    duration: float = 300.0
    fs: float = 16_000.0
    t0: float = 0.0
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    ground: Ground = field(default_factory=Ground)
    ambient: AmbientConfig | None = field(default_factory=AmbientConfig)
    texture: NoiseProfile | None = None  # if set, replaces the parametric background
    device: Device = field(default_factory=Device)
    seed: int = 0


@dataclass
class SimResult:
    audio: F64  # digital samples in [-1, 1] after the device chain
    fs: float
    t0: float
    components: dict[str, F64]  # pressure (Pa) per component, before the device
    truth: list[EventTruth]
    rendered: list[RenderedSource]
    scenario: Scenario


def _truth(ap: AircraftPass, rs: RenderedSource) -> EventTruth:
    d = rs.direct
    i = int(np.argmin(np.where(d.valid, d.rng, np.inf)))
    pos, vel = ap.traj.state(np.array([d.t_e[i]]))
    return EventTruth(
        label=ap.label or ap.source.name,
        aircraft_class=ap.source.aircraft_class,
        t_cpa_source=float(d.t_e[i]),
        t_cpa_heard=float(rs.control_t[i]),
        r_cpa=float(d.rng[i]),
        speed_cpa=float(np.linalg.norm(vel[0, :2])),
        alt_cpa=float(pos[0, 2]),
        f0=ap.source.fundamental_hz,
        control_t=rs.control_t,
        tone_freqs=rs.received_tone_freqs(),
        tone_levels_db=np.array([t.level_db for t in ap.source.tones]),
        active=rs.active,
        range_m=d.rng,
    )


def simulate(scn: Scenario) -> SimResult:
    rng = np.random.default_rng(scn.seed)
    n = int(round(scn.duration * scn.fs))
    comps: dict[str, F64] = {}
    rendered, truth = [], []
    aircraft_sum = np.zeros(n)
    for k, ap in enumerate(scn.aircraft):
        rs = render_source(ap.traj, ap.source, scn.fs, scn.t0, scn.duration, scn.atmosphere, scn.ground, rng)
        aircraft_sum += rs.signal
        comps[f"aircraft_{k}"] = rs.signal
        rendered.append(rs)
        truth.append(_truth(ap, rs))
    comps["aircraft"] = aircraft_sum

    total = aircraft_sum.copy()
    if scn.texture is not None:
        comps["texture"] = texture_noise(scn.texture, scn.fs, scn.duration, rng)
        total += comps["texture"]
    if scn.ambient is not None:
        amb = render_ambient(scn.ambient, scn.fs, scn.duration, rng)
        if scn.texture is not None:
            amb.pop("background", None)
        for k, v in amb.items():
            if k.startswith("_"):
                comps[k] = v
                continue
            comps[k] = v
            total += v
    comps["total_pa"] = total
    audio = apply_device(total, scn.fs, scn.device, rng)
    return SimResult(audio, scn.fs, scn.t0, comps, truth, rendered, scn)


# ---- convenience generators -------------------------------------------------------


def straight_pass(
    preset: str | SourceModel,
    t_cpa_source: float,
    r_cpa: float,
    alt: float,
    speed: float,
    heading_deg: float,
    side: float = 1.0,
    half_span_s: float = 600.0,
    label: str = "",
) -> AircraftPass:
    """Level straight pass; mic at the origin. `side` (+1/-1) picks which side of the mic."""
    src = PRESETS[preset] if isinstance(preset, str) else preset
    horiz = np.sqrt(max(r_cpa**2 - alt**2, 0.0))
    h = np.radians(heading_deg)
    vel = np.array([speed * np.sin(h), speed * np.cos(h), 0.0])
    perp = side * np.array([np.cos(h), -np.sin(h), 0.0])
    traj = Trajectory.straight(
        perp * horiz + [0.0, 0.0, alt],
        vel,
        t_cpa_source,
        t_cpa_source - half_span_s,
        t_cpa_source + half_span_s,
        dt=1.0,
    )
    return AircraftPass(traj, src, label or (src.meta.get("icao_type", src.name)))


def manoeuvring_pass(
    preset: str | SourceModel,
    t_start: float,
    p0: tuple[float, float, float],
    duration: float,
    speed: F64 | float,
    heading_deg: F64 | float,
    vrate: F64 | float = 0.0,
    dt: float = 1.0,
    label: str = "",
) -> AircraftPass:
    """Arbitrary kinematic path; speed/heading/vrate may be arrays sampled every `dt`."""
    src = PRESETS[preset] if isinstance(preset, str) else preset
    t = t_start + np.arange(0, duration + dt, dt)
    traj = Trajectory.from_kinematics(t, p0, speed, heading_deg, vrate)
    return AircraftPass(traj, src, label or (src.meta.get("icao_type", src.name)))
