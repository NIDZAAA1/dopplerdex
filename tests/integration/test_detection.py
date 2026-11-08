"""End-to-end detection on synthetic recordings with known truth (seconds per case)."""

import numpy as np
import pytest

from dopplerdex.audio.events import DetectConfig, detect_events, fit_doppler
from dopplerdex.audio.spectrogram import resample
from dopplerdex.physics.doppler import straight_pass_frequency
from dopplerdex.sim.noise import AmbientConfig
from dopplerdex.sim.scenario import Scenario, simulate, straight_pass


def _detect(scn: Scenario):
    res = simulate(scn)
    x = resample(res.audio, int(scn.fs), 8000)
    ev, _ = detect_events(x, 8000, DetectConfig(c=scn.atmosphere.c))
    return res, ev


def test_fit_recovers_parameters_from_clean_track():
    t = np.linspace(0, 200, 800)
    c = 343.0
    f = straight_pass_frequency(t, 112.0, 95.0, 2500.0, 92.0, c) * (
        1 + 0.002 * np.random.default_rng(0).normal(size=t.size)
    )
    fit = fit_doppler(t, f, c=c)
    s = fit.sigma()
    assert abs(fit.v - 95.0) < 3 * s["v"] + 0.5
    assert abs(fit.r_cpa - 2500.0) < 3 * s["r_cpa"] + 10
    assert abs(fit.t_cpa_heard - (92.0 + 2500.0 / c)) < 3 * s["t_cpa_heard"] + 0.1


@pytest.mark.parametrize(
    "preset,r,alt,v",
    [("C172", 1200.0, 600.0, 55.0), ("BE20", 3000.0, 1500.0, 100.0), ("R44", 900.0, 300.0, 45.0)],
)
def test_single_pass_detected_with_accurate_parameters(preset, r, alt, v):
    scn = Scenario([straight_pass(preset, 95.0, r, alt, v, 60.0)], duration=200.0, seed=11)
    res, ev = _detect(scn)
    tr = res.truth[0]
    assert len(ev) == 1, [(e.fit.t_cpa_heard, e.fit.v) for e in ev]
    f = ev[0].fit
    s = f.sigma()
    assert abs(f.t_cpa_heard - tr.t_cpa_heard) < max(3 * s["t_cpa_heard"], 1.5)
    assert abs(f.v - tr.speed_cpa) < max(3 * s["v"], 0.05 * tr.speed_cpa)
    assert abs(f.r_cpa - tr.r_cpa) < max(3 * s["r_cpa"], 0.15 * tr.r_cpa)


def test_two_events_offset_recovered():
    scn = Scenario(
        [straight_pass("SR22", 50.0, 1800, 900, 75, 20), straight_pass("C208", 250.0, 2500, 1200, 85, 200, side=-1)],
        duration=330.0,
        seed=6,
    )
    res, ev = _detect(scn)
    assert len(ev) == 2
    dt_true = res.truth[1].t_cpa_heard - res.truth[0].t_cpa_heard
    dt_est = ev[1].fit.t_cpa_heard - ev[0].fit.t_cpa_heard
    assert abs(dt_est - dt_true) < 1.5


def test_no_events_in_ambient_noise_only():
    amb = AmbientConfig(background_db=48, steady_tones=((118.0, 40.0), (241.0, 36.0), (400.0, 35.0)), birds_per_min=6,
                        transients_per_min=4, wind_db=55)  # fmt: skip
    scn = Scenario([], duration=300.0, ambient=amb, seed=3)
    _, ev = _detect(scn)
    assert ev == []


def test_ground_traffic_classification():
    from dopplerdex.audio.events import DetectedEvent, DopplerFit, classify_ground
    from dopplerdex.audio.tracking import Ridge

    def ev(v, r):
        fit = DopplerFit(f0=100.0, v=v, r_cpa=r, tau=0.0, c=343.0, cov=np.eye(4))
        z = np.zeros(3)
        return DetectedEvent(Ridge(z, z, z + 100, z, 0.0), fit, 0.1)

    events = [ev(30.0, 120.0), ev(28.0, 350.0), ev(55.0, 2400.0), ev(40.0, 900.0), ev(90.0, 300.0)]
    classify_ground(events, DetectConfig())
    assert [e.tier for e in events] == ["G", "G", "A", "A", "A"]
