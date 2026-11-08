"""Ground-reflection (Lloyd's mirror) delay events: recovered from a rendered jet pass, absent in noise."""

import numpy as np

from dopplerdex.audio.multipath import detect_multipath
from dopplerdex.physics.trajectory import Trajectory
from dopplerdex.sim.render import Atmosphere, Ground, render_source
from dopplerdex.sim.sources import PRESETS

FS = 8000


def test_jet_pass_recovered():
    rng = np.random.default_rng(1)
    H, side, v, h = 900.0, 600.0, 80.0, 1.5
    tr = Trajectory.straight([side, 0.0, H], [0.0, v, 0.0], 100.0, 0.0, 200.0, 0.5)
    x = render_source(tr, PRESETS["A320"], FS, 0.0, 200.0, Atmosphere(), Ground(mic_height=h), rng).signal
    x = x / np.std(x) * 0.1 + rng.normal(0, 0.03, x.size)
    ev = detect_multipath(x, FS)
    assert len(ev) == 1
    S0 = np.hypot(H, side)
    e = ev[0]
    assert abs(e.T - S0 / v) / (S0 / v) < 0.1
    assert abs(e.tau_max - 2 * h * H / S0 / 340.0) / (2 * h * H / S0 / 340.0) < 0.05
    assert abs(e.t_cpa - (100.0 + S0 / 340.0)) < 1.5


def test_noise_has_no_events():
    rng = np.random.default_rng(2)
    assert detect_multipath(rng.normal(0, 0.03, 300 * FS), FS) == []
