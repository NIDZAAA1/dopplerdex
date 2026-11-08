import numpy as np

from dopplerdex.adsb.store import Flight
from dopplerdex.physics.trajectory import Trajectory
from dopplerdex.search.loudness import align_by_loudness, predicted_level
from dopplerdex.search.oracle import predict_passes


def _flight(t_cpa: float, r_side: float, cls: str = "jet") -> Flight:
    tr = Trajectory.straight([r_side, 0.0, 600.0], [0.0, 90.0, 0.0], t_cpa, t_cpa - 300, t_cpa + 300, 1.0)
    return Flight("abc", "X", cls, 0, tr, "geom", tr.t.size, 1.0)


def test_predict_passes_cpa():
    ps = predict_passes([_flight(1000.0, 800.0)], np.zeros(3), 343.0)
    assert len(ps) == 1
    p = ps[0]
    assert abs(p.r_cpa - 1000.0) < 1.0  # sqrt(800^2 + 600^2)
    assert abs(p.t_cpa_heard - (1000.0 + 1000.0 / 343.0)) < 0.3


def test_loudness_alignment_recovers_offset():
    rng = np.random.default_rng(0)
    cpas = np.sort(rng.uniform(0, 20000, 60))
    flights = [_flight(t, rng.uniform(500, 4000)) for t in cpas]
    ps = predict_passes(flights, np.zeros(3), 343.0, max_range=12000)
    T0_true = 7300.0
    t_rec = np.arange(0, 3600.0, 1.0)
    obs = predicted_level(ps, T0_true + t_rec) + rng.normal(0, 2.0, t_rec.size)  # noisy observation
    T0, _, corr = align_by_loudness(t_rec, obs, ps, 0.0, 16000.0)
    assert abs(T0 - T0_true) <= 2.0
