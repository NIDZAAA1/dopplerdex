"""Forward-model every flight near a hypothesised location: what would a mic there have heard?

This is the core "forward model" of the locator, used here for oracle validation (true location
known) and later inside fine scoring (candidate locations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dopplerdex.adsb.store import Flight
from dopplerdex.physics.doppler import solve_retarded

F64 = NDArray[np.float64]


@dataclass
class PredictedPass:
    flight: Flight
    t_r: F64 = field(repr=False)  # reception times (unix s)
    doppler: F64 = field(repr=False)  # received/emitted frequency ratio
    rng: F64 = field(repr=False)
    valid: NDArray[np.bool_] = field(repr=False)
    t_cpa_heard: float = np.nan
    r_cpa: float = np.nan
    speed_cpa: float = np.nan
    alt_cpa: float = np.nan  # above the mic (m)
    sin_el: F64 | None = field(default=None, repr=False)  # sin(source elevation) at the mic, per t_r


def predict_passes(
    flights: list[Flight],
    mic_enu: F64,
    c: float,
    wind: F64 | None = None,
    dt: float = 0.25,
    max_range: float = 15_000.0,
) -> list[PredictedPass]:
    w = np.zeros(3) if wind is None else np.asarray(wind, dtype=np.float64)
    out = []
    for fl in flights:
        tr = fl.traj
        # quick reject: closest sampled point
        d0 = np.linalg.norm(tr.pos - mic_enu, axis=1)
        if d0.min() > max_range:
            continue
        t_r = np.arange(tr.t_start + d0[0] / c, tr.t_end + d0[-1] / c, dt)
        if t_r.size < 4:
            continue
        sol = solve_retarded(tr, mic_enu, t_r, c, w)
        m = sol.valid & (sol.rng < max_range)
        if not m.any():
            continue
        i = int(np.argmin(np.where(m, sol.rng, np.inf)))
        out.append(
            PredictedPass(
                flight=fl,
                t_r=t_r,
                doppler=sol.doppler,
                rng=sol.rng,
                valid=m,
                t_cpa_heard=float(t_r[i]),
                r_cpa=float(sol.rng[i]),
                speed_cpa=float(np.linalg.norm(sol.src_vel[i, :2])),
                alt_cpa=float(sol.src_pos[i, 2] - mic_enu[2]),
                sin_el=(sol.src_pos[:, 2] - mic_enu[2]) / np.maximum(sol.rng, 1.0),
            )
        )
    out.sort(key=lambda p: p.t_cpa_heard)
    return out
