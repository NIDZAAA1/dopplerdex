"""Coarse candidate geometry, pair join and ridge-match scoring on synthetic tracks with known truth."""

import numpy as np
import polars as pl

from dopplerdex.audio.tracking import Ridge
from dopplerdex.physics.doppler import straight_pass_frequency
from dopplerdex.scoring.ridge_match import RidgeSet, score_hypotheses
from dopplerdex.search.coarse import EventConstraint, generate, join_pairs
from dopplerdex.search.tracks import TrackSamples

C = 343.0
LAT0, LON0 = 40.0, -100.0


def _track(fid, t_cpa, side_m, alt, v, heading_deg, dt=6.0, span=900.0):
    """Straight level track passing the mic at (LAT0, LON0) with horizontal CPA offset side_m."""
    h = np.radians(heading_deg)
    t = np.arange(t_cpa - span, t_cpa + span, dt)
    along = (t - t_cpa) * v
    e = along * np.sin(h) + side_m * np.cos(h)
    n = along * np.cos(h) - side_m * np.sin(h)
    lat = LAT0 + n / 111320.0
    lon = LON0 + e / (111320.0 * np.cos(np.radians(LAT0)))
    return dict(fid=np.full(t.size, fid, np.int32), t=t, lat=lat, lon=lon, alt=np.full(t.size, alt),
                gs=np.full(t.size, v), trk=np.full(t.size, heading_deg % 360))  # fmt: skip


def _samples(*tracks):
    cat = {k: np.concatenate([tr[k] for tr in tracks]) for k in tracks[0]}
    flights = pl.DataFrame({"id": [int(tr["fid"][0]) for tr in tracks]})
    return TrackSamples(cat["fid"], cat["t"], cat["lat"], cat["lon"], cat["alt"], cat["gs"], cat["trk"],
                        np.ones(cat["t"].size, np.int8), flights)  # fmt: skip


def _event(t_heard, v, r, sv=5.0, sr=300.0, st=1.0):
    return EventConstraint.from_values(v, sv, r, sr, t_heard, st, rho_vr=0.9, r_margin=200.0)


def test_generate_contains_truth_and_correct_t0():
    T0 = 50_000.0
    alt, side, v = 1000.0, 2000.0, 80.0
    r_true = np.hypot(alt, side)
    ts = _samples(_track(0, T0 + 600.0, side, alt, v, 30.0))
    ev = _event(600.0 + r_true / C, v, r_true)
    cand = generate(ts, np.zeros(len(ts)), ev, C, step=100.0, cell_m=500.0, t0_range=(T0 - 3600, T0 + 3600))
    d = np.hypot((cand.lat - LAT0) * 111320, (cand.lon - LON0) * 111320 * np.cos(np.radians(LAT0)))
    near = d < 250
    assert near.any()
    assert np.min(np.abs(cand.t0[near] - T0)) < 3.0


def test_join_finds_true_pair_location():
    T0 = 80_000.0
    a = _track(0, T0 + 300.0, 1500.0, 900.0, 70.0, 10.0)
    b = _track(1, T0 + 1200.0, -2500.0, 1200.0, 100.0, 250.0)
    ts = _samples(a, b)
    ea = _event(300.0 + np.hypot(1500, 900) / C, 70.0, np.hypot(1500, 900))
    eb = _event(1200.0 + np.hypot(2500, 1200) / C, 100.0, np.hypot(2500, 1200))
    rng = (T0 - 7200, T0 + 7200)
    A = generate(ts, np.zeros(len(ts)), ea, C, 200.0, 1000.0, rng)
    B = generate(ts, np.zeros(len(ts)), eb, C, 200.0, 1000.0, rng)
    pm = join_pairs(A, B, tol=8.0)
    assert len(pm) > 0
    lat = 0.5 * (A.lat[pm.a_idx] + B.lat[pm.b_idx])
    lon = 0.5 * (A.lon[pm.a_idx] + B.lon[pm.b_idx])
    d = np.hypot((lat - LAT0) * 111320, (lon - LON0) * 111320 * np.cos(np.radians(LAT0)))
    assert d.min() < 800
    assert (A.fid[pm.a_idx] != B.fid[pm.b_idx]).all()


def _ridge_for(t_rec, f0, v, r, t_cpa_src):
    f = straight_pass_frequency(t_rec, f0, v, r, t_cpa_src, C)
    return Ridge(t_rec, np.zeros_like(t_rec), f, np.full(t_rec.size, 8.0), 8.0)


def test_ridge_match_scores_truth_best():
    T0 = 30_000.0
    ta = _track(0, T0 + 400.0, 1800.0, 1000.0, 75.0, 40.0, dt=2.0)
    tb = _track(1, T0 + 1300.0, -2200.0, 800.0, 95.0, 200.0, dt=2.0)
    ts = _samples(ta, tb)
    ra, rb = np.hypot(1800, 1000), np.hypot(2200, 800)
    # ridges as the detector would deliver them (recording time)
    t_a = np.linspace(400 - 60, 400 + 80, 120)
    t_b = np.linspace(1300 - 60, 1300 + 80, 120)

    class Ev:  # minimal DetectedEvent stand-in
        def __init__(self, ridge):
            self.ridge, self.companions = ridge, []

    rsa = RidgeSet.from_event(Ev(_ridge_for(t_a, 110.0, 75.0, ra, 400.0)))
    rsb = RidgeSet.from_event(Ev(_ridge_for(t_b, 130.0, 95.0, rb, 1300.0)))
    fid = ts.fid
    starts = np.searchsorted(fid, [0, 1]).astype(np.int64)
    ends = np.searchsorted(fid, [0, 1], side="right").astype(np.int64)
    # truth and four displaced hypotheses (1 km N/S/E/W)
    dlat = np.array([0, 1000, -1000, 0, 0]) / 111320.0
    dlon = np.array([0, 0, 0, 1000, -1000]) / (111320.0 * np.cos(np.radians(LAT0)))
    n = dlat.size
    rms_a, rms_b, _ = score_hypotheses(
        LAT0 + dlat, LON0 + dlon, np.zeros(n), np.full(n, T0), np.zeros(n, np.int64), np.ones(n, np.int64),
        ts.t, ts.lat, ts.lon, ts.alt_msl, ts.gs, ts.trk, np.zeros(len(ts)), starts, ends,
        rsa.t, rsa.lf, rsa.w, rsa.seg, rsa.n_seg, rsb.t, rsb.lf, rsb.w, rsb.seg, rsb.n_seg,
        C, np.arange(-4.0, 4.01, 0.5), 0.03, 0.6,
    )  # fmt: skip
    total = rms_a + rms_b
    assert np.argmin(total) == 0
    assert total[0] < 2.0  # cents: geometry reproduces the curves almost exactly
    assert total[1:].min() > 3 * total[0]


def test_loud_events_find_broadband_passes():
    from dopplerdex.audio.loud_events import detect_loud_events

    fs = 8000.0
    rng = np.random.default_rng(0)
    t = np.arange(int(900 * fs)) / fs
    x = 0.01 * rng.normal(size=t.size)
    for tc in (200.0, 520.0):  # two 30 s broadband swells, +20 dB
        x += 0.1 * rng.normal(size=t.size) * np.exp(-0.5 * ((t - tc) / 10.0) ** 2)
    ev = detect_loud_events(x, fs)
    got = sorted(e.t_peak for e in ev)
    assert len(got) == 2 and abs(got[0] - 200) < 4 and abs(got[1] - 520) < 4
