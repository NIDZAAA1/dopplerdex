"""Synthetic recordings rendered from REAL ADS-B tracks at a chosen place and time.

This gives end-to-end test cases with exact ground truth that exercise the whole pipeline on real
traffic patterns (airport approaches, crossing traffic, silence gaps), not just idealised passes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dopplerdex.adsb.store import Flight, build_flights, load_points
from dopplerdex.env.dem import fine_elevation
from dopplerdex.env.geoid import undulation
from dopplerdex.geo.frames import LocalFrame
from dopplerdex.physics.trajectory import Trajectory
from dopplerdex.sim.noise import AmbientConfig
from dopplerdex.sim.render import Atmosphere, Device, Ground
from dopplerdex.sim.scenario import AircraftPass, Scenario
from dopplerdex.sim.sources import PRESETS

TYPE_TO_PRESET = {"C172": "C172", "C152": "C172", "P28A": "PA28", "P28R": "PA28", "PA28": "PA28", "SR22": "SR22",
                  "SR20": "SR22", "S22T": "SR22", "BE20": "BE20", "B350": "BE20", "BE9L": "BE20", "C208": "C208",
                  "PC12": "C208", "DH8D": "DH8D", "AT76": "DH8D", "R44": "R44", "R22": "R44"}  # fmt: skip
CLASS_TO_PRESETS = {"piston": ("C172", "PA28", "SR22"), "turboprop": ("BE20", "C208", "DH8D"), "helicopter": ("R44",),
                    "jet": ("A320",), "unknown": ("C172", "PA28"), "other": ("C172",)}  # fmt: skip


@dataclass
class RealScenario:
    scenario: Scenario
    lat: float
    lon: float
    t0: float  # recording start (unix s)
    ground_msl: float
    flights: list[Flight]


def preset_for(fl: Flight, rng: np.random.Generator) -> str:
    if fl.type in TYPE_TO_PRESET:
        return TYPE_TO_PRESET[fl.type]
    opts = CLASS_TO_PRESETS.get(fl.aircraft_class, ("C172",))
    return str(opts[int(rng.integers(0, len(opts)))])


def build_real_scenario(lat: float, lon: float, t0: float, duration: float, seed: int, fs: float = 16_000.0,
                        radius_km: float = 20.0, mic_height: float = 2.0, include_jets: bool = True,
                        ambient: AmbientConfig | None = None, region: str = "north_america") -> RealScenario:  # fmt: skip
    rng = np.random.default_rng(seed)
    ground_msl = float(fine_elevation(lat, lon)[0])
    n = float(undulation(lat, lon))
    frame = LocalFrame(lat, lon, 0.0)
    df = load_points(t0 - 900, t0 + duration + 900, lat, lon, radius_km, region)
    flights = build_flights(df, frame, geoid_n=n)
    ground_hae = ground_msl + n
    passes = []
    for fl in flights:
        if fl.aircraft_class == "jet" and not include_jets:
            continue
        d = np.linalg.norm(fl.traj.pos[:, :2], axis=1)
        if d.min() > radius_km * 1000:
            continue
        pos = fl.traj.pos.copy()
        pos[:, 2] -= ground_hae  # render frame: local ground plane at z = 0, mic at z = mic_height
        traj = Trajectory(fl.traj.t, pos, fl.traj.vel)
        passes.append(
            AircraftPass(traj, PRESETS[preset_for(fl, rng)], label=f"{fl.icao}:{fl.type or fl.aircraft_class}")
        )
    scn = Scenario(
        passes,
        duration=duration,
        fs=fs,
        t0=t0,
        atmosphere=Atmosphere(
            temp_c=float(rng.uniform(5, 30)),
            rel_humidity=float(rng.uniform(30, 90)),
            turbulence_db=float(rng.uniform(0.5, 2.5)),
        ),  # fmt: skip
        ground=Ground(mic_height=mic_height, flow_resistivity=float(rng.choice([200e3, 300e3, 2e6]))),
        ambient=ambient
        or AmbientConfig(
            background_db=float(rng.uniform(36, 50)),
            wind_db=float(rng.uniform(30, 52)),
            birds_per_min=float(rng.uniform(0, 4)),
        ),  # fmt: skip
        device=Device(full_scale_db=float(rng.uniform(100, 112))),
        seed=seed,
    )
    return RealScenario(scn, lat, lon, t0, ground_msl, flights)


def with_duration(rs: RealScenario, duration: float) -> RealScenario:
    return replace(rs, scenario=replace(rs.scenario, duration=duration))
