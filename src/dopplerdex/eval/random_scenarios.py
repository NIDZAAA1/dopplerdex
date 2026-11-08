"""Random synthetic scenarios with ground truth, for Monte-Carlo benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dopplerdex.sim.catalog import challenge_texture
from dopplerdex.sim.noise import AmbientConfig
from dopplerdex.sim.render import Atmosphere, Device, Ground
from dopplerdex.sim.scenario import AircraftPass, Scenario, straight_pass

# (preset, typical speed range m/s, altitude range m)
CLASSES = {
    "C172": ((45, 65), (200, 1800)),
    "PA28": ((45, 65), (200, 1800)),
    "SR22": ((60, 90), (300, 2500)),
    "BE20": ((80, 130), (400, 3500)),
    "C208": ((65, 95), (300, 3000)),
    "DH8D": ((90, 140), (600, 3500)),
    "R44": ((30, 55), (150, 900)),
}


@dataclass(frozen=True)
class RandomScenarioSpec:
    duration: float = 240.0
    fs: float = 16_000.0
    n_aircraft: tuple[int, int] = (1, 2)
    r_cpa: tuple[float, float] = (300.0, 6000.0)
    codecs: tuple[str | None, ...] = (None, None, None, "mp3_64k", "phone_nb")
    texture_prob: float = 0.35


def random_scenario(seed: int, spec: RandomScenarioSpec = RandomScenarioSpec()) -> tuple[Scenario, str | None]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(spec.n_aircraft[0], spec.n_aircraft[1] + 1))
    passes: list[AircraftPass] = []
    # keep heard CPAs separated so that "which event is which" is unambiguous in scoring
    slots = np.sort(rng.choice(np.arange(40, spec.duration - 30, 50.0), size=n, replace=False))
    for t_slot in slots:
        name = str(rng.choice(list(CLASSES)))
        (v_lo, v_hi), (a_lo, a_hi) = CLASSES[name]
        r = float(np.exp(rng.uniform(np.log(spec.r_cpa[0]), np.log(spec.r_cpa[1]))))
        alt = float(min(rng.uniform(a_lo, a_hi), 0.98 * r))
        v = float(rng.uniform(v_lo, v_hi))
        t_cpa = float(t_slot + rng.uniform(-10, 10) - r / 340.0)
        passes.append(straight_pass(name, t_cpa, r, alt, v, float(rng.uniform(0, 360)), side=float(rng.choice([-1, 1])),
                                    half_span_s=max(600.0, 3 * spec.duration)))  # fmt: skip
    dev = Device(full_scale_db=float(rng.uniform(100, 115)))
    use_tex = rng.random() < spec.texture_prob
    tex = challenge_texture(dev) if use_tex else None
    amb = AmbientConfig(
        background_db=float(rng.uniform(35, 55)),
        background_tilt_db_oct=float(rng.uniform(-6, -2)),
        wind_db=float(rng.uniform(30, 60)),
        mains_hz=float(rng.choice([50.0, 60.0])),
        mains_db=float(rng.uniform(15, 40)),
        steady_tones=tuple(
            (float(rng.uniform(80, 900)), float(rng.uniform(20, 45))) for _ in range(rng.integers(0, 4))
        ),
        birds_per_min=float(rng.uniform(0, 6)),
        transients_per_min=float(rng.uniform(0, 4)),
    )
    wind_dir = rng.uniform(0, 2 * np.pi)
    ws = float(rng.uniform(0, 10))
    scn = Scenario(
        passes,
        duration=spec.duration,
        fs=spec.fs,
        atmosphere=Atmosphere(
            temp_c=float(rng.uniform(-5, 35)),
            rel_humidity=float(rng.uniform(20, 95)),
            wind=(ws * np.sin(wind_dir), ws * np.cos(wind_dir), 0.0),
            turbulence_db=float(rng.uniform(0.5, 3.0)),
        ),
        ground=Ground(
            mic_height=float(rng.uniform(0.5, 6.0)), flow_resistivity=float(rng.choice([150e3, 300e3, 2e6, 2e7]))
        ),
        ambient=amb,
        texture=tex,
        device=dev,
        seed=seed,
    )
    codec = spec.codecs[int(rng.integers(0, len(spec.codecs)))]
    return scn, codec
