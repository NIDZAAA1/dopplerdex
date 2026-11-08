import numpy as np
import pytest

from dopplerdex.geo.frames import LocalFrame, ecef_to_geodetic, geodetic_to_ecef, velocity_enu
from dopplerdex.physics.atmosphere import absorption_db_per_m, speed_of_sound
from dopplerdex.physics.doppler import (
    naive_doppler,
    propagation_delay,
    solve_retarded,
    straight_pass_emission_offset,
    straight_pass_frequency,
    straight_pass_params_from_plateaus,
)
from dopplerdex.physics.propagation import (
    image_observer,
    lloyd_mirror_null_frequencies,
    plane_wave_reflection,
)
from dopplerdex.physics.trajectory import Trajectory

C = 343.0


def level_pass(v=60.0, r_cpa=1500.0, alt=1000.0, t_cpa=100.0, heading_deg=90.0):
    """Level straight pass; observer at origin; CPA slant distance r_cpa."""
    horiz = np.sqrt(r_cpa**2 - alt**2)
    h = np.radians(heading_deg)
    vel = np.array([v * np.sin(h), v * np.cos(h), 0.0])
    side = np.array([np.cos(h), -np.sin(h), 0.0])  # perpendicular to the track
    p_cpa = side * horiz + np.array([0, 0, alt])
    return Trajectory.straight(p_cpa, vel, t_cpa, t_cpa - 400, t_cpa + 400, dt=0.5)


# ---------------- frames ----------------


def test_ecef_roundtrip():
    lat, lon, h = np.array([40.8765, -33.9, 89.9]), np.array([-73.525, 151.2, 10.0]), np.array([12.0, 500.0, -30.0])
    la2, lo2, h2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(la2, lat, atol=1e-9)
    np.testing.assert_allclose(lo2, lon, atol=1e-9)
    np.testing.assert_allclose(h2, h, atol=1e-4)


def test_local_frame_up_and_distance():
    f = LocalFrame(40.0, -100.0, 0.0)
    enu = f.to_enu(40.0, -100.0, 1000.0)
    np.testing.assert_allclose(enu, [0, 0, 1000.0], atol=1e-6)
    # 0.01 deg north ~ 1111 m
    e = f.to_enu(40.01, -100.0, 0.0)
    assert abs(e[1] - 1111.0) < 3.0 and abs(e[0]) < 1e-6
    la, lo, h = f.to_geodetic(e)
    assert abs(la - 40.01) < 1e-10 and abs(lo + 100.0) < 1e-10 and abs(h) < 1e-6


def test_velocity_enu_track_convention():
    v = velocity_enu(100.0, 90.0, 0.0)  # due east
    np.testing.assert_allclose(v, [100.0, 0.0, 0.0], atol=1e-12)


# ---------------- atmosphere ----------------


def test_speed_of_sound():
    assert abs(speed_of_sound(20.0) - 343.4) < 0.6
    assert speed_of_sound(30.0) > speed_of_sound(0.0)


@pytest.mark.parametrize(
    "f,expected_db_km",
    # ISO 9613-2 Table 2, 20 C / 70 % RH (values rounded in the standard)
    [(63, 0.1), (125, 0.3), (250, 1.1), (500, 2.8), (1000, 5.0), (2000, 9.0), (4000, 22.9), (8000, 76.6)],
)
def test_absorption_matches_iso_table(f, expected_db_km):
    a = absorption_db_per_m(f, 20.0, 70.0) * 1000
    assert abs(a - expected_db_km) <= max(0.06, 0.04 * expected_db_km)


# ---------------- trajectory ----------------


def test_hermite_reproduces_constant_velocity_exactly():
    tr = level_pass()
    tq = np.linspace(tr.t_start, tr.t_end, 777)
    pos, vel = tr.state(tq)
    exp = tr.pos[0] + (tq - tr.t[0])[:, None] * tr.vel[0]
    np.testing.assert_allclose(pos, exp, atol=1e-6)
    np.testing.assert_allclose(vel, np.broadcast_to(tr.vel[0], vel.shape), atol=1e-9)


def test_from_kinematics_turn_is_smooth():
    t = np.arange(0, 120, 1.0)
    hdg = np.where(t < 60, 0.0, np.minimum((t - 60) * 3.0, 90.0))  # standard-rate turn
    tr = Trajectory.from_kinematics(t, [0, 0, 1000], 60.0, hdg, 0.0)
    _, v = tr.state(np.linspace(0, 119, 1000))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 60.0, rtol=1e-3)


# ---------------- Doppler ----------------


def test_delay_still_air_is_range_over_c():
    d = np.array([[300.0, 400.0, 1200.0]])
    assert abs(propagation_delay(d, C)[0] - 1300.0 / C) < 1e-12


def test_delay_with_wind_tailwind_shorter():
    d = np.array([[1000.0, 0.0, 0.0]])  # observer downwind (east) of source
    assert propagation_delay(d, C, [10.0, 0, 0])[0] == pytest.approx(1000.0 / (C + 10.0))
    assert propagation_delay(d, C, [-10.0, 0, 0])[0] == pytest.approx(1000.0 / (C - 10.0))


def test_retarded_matches_closed_form():
    v, r, tc, f0 = 60.0, 1500.0, 100.0, 90.0
    tr = level_pass(v=v, r_cpa=r, t_cpa=tc)
    t_r = np.linspace(-200, 400, 2001)
    sol = solve_retarded(tr, [0, 0, 0], t_r, C)
    assert sol.max_residual < 1e-8
    assert sol.iterations <= 6
    f_closed = straight_pass_frequency(t_r, f0, v, r, tc, C)
    np.testing.assert_allclose(f0 * sol.doppler, f_closed, rtol=1e-9)
    u = straight_pass_emission_offset(t_r - tc, v, r, C)
    np.testing.assert_allclose(sol.t_e - tc, u, atol=1e-7)


def test_f0_heard_exactly_when_cpa_sound_arrives():
    v, r, tc, f0 = 80.0, 2000.0, 50.0, 120.0
    t_hear = tc + r / C
    assert straight_pass_frequency(t_hear, f0, v, r, tc, C) == pytest.approx(f0, rel=1e-12)


def test_plateaus_and_inversion():
    v, f0 = 90.0, 110.0
    fa, fr = f0 / (1 - v / C), f0 / (1 + v / C)
    f_far = straight_pass_frequency(np.array([-1e7, 1e7]), f0, v, 1000.0, 0.0, C)
    np.testing.assert_allclose(f_far, [fa, fr], rtol=1e-6)
    f0_hat, v_hat = straight_pass_params_from_plateaus(fa, fr, C)
    assert f0_hat == pytest.approx(f0) and v_hat == pytest.approx(v)


def test_naive_model_differs_from_retarded():
    """The video's 'wrong model': ignoring retardation gives a different curve, not a shifted one."""
    tr = level_pass(v=100.0, r_cpa=3000.0, alt=1000.0, t_cpa=0.0)
    t_r = np.linspace(-60, 80, 500)
    right = solve_retarded(tr, [0, 0, 0], t_r, C).doppler
    wrong = naive_doppler(tr, [0, 0, 0], t_r, C)
    assert np.max(np.abs(right - wrong)) > 0.05
    # best pure time-shift of the wrong curve still leaves a large residual
    best = min(np.max(np.abs(np.interp(t_r, t_r + s, wrong) - right)[50:-50]) for s in np.arange(0, 15, 0.1))
    assert best > 0.005


def test_wind_doppler_still_air_limit_and_symmetry():
    tr = level_pass()
    t_r = np.linspace(0, 200, 300)
    a = solve_retarded(tr, [0, 0, 0], t_r, C, wind=[0, 0, 0]).doppler
    b = solve_retarded(tr, [0, 0, 0], t_r, C, wind=[1e-9, 0, 0]).doppler
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_wind_doppler_matches_air_frame_galilean_check():
    """In uniform wind the received *frequency* depends on the source velocity relative to the
    air and on the observer's velocity relative to the air (-w). Verify against the classic
    moving-observer / moving-source formula evaluated in the air frame."""
    tr = level_pass(v=70.0, r_cpa=1200.0, alt=800.0, heading_deg=30.0)
    w = np.array([8.0, -5.0, 0.0])
    t_r = np.linspace(-50, 300, 400)
    sol = solve_retarded(tr, [0, 0, 0], t_r, C, wind=w)
    n = sol.n  # wavefront normal (propagation direction in the air frame)
    vs_air = sol.src_vel - w
    vo_air = -w
    f_air = (C - n @ vo_air) / (C - np.einsum("ij,ij->i", n, vs_air))
    np.testing.assert_allclose(sol.doppler, f_air, rtol=1e-10)


# ---------------- propagation ----------------


def test_image_observer_path_length():
    o = np.array([0.0, 0.0, 1.5])
    s = np.array([500.0, 200.0, 1000.0])
    oi = image_observer(o, 0.0)
    assert oi[2] == -1.5
    direct, refl = np.linalg.norm(s - o), np.linalg.norm(s - oi)
    assert refl - direct == pytest.approx(2 * 1.5 * (s[2] / np.linalg.norm(s)), rel=2e-3)


def test_rigid_reflection_is_unity_and_soft_ground_is_lossy():
    assert np.all(plane_wave_reflection(np.array([100.0, 500.0]), 0.3, None) == 1)
    r = plane_wave_reflection(np.array([100.0, 1000.0]), np.radians(5), 200e3)
    assert np.all(np.abs(r) < 1.0)


def test_lloyd_nulls():
    nulls = lloyd_mirror_null_frequencies(1.5, np.pi / 2, 343.0, 3)
    np.testing.assert_allclose(nulls, [57.17, 171.5, 285.8], rtol=2e-3)


def test_extrapolation_consistent_and_flagged():
    tr = level_pass(v=100.0, t_cpa=0.0)  # span [-400, 400]
    p, v = tr.state(np.array([-500.0, 500.0]))
    np.testing.assert_allclose(p[1] - p[0], 1000.0 * v[0], atol=1e-6)
    sol = solve_retarded(tr, [0, 0, 0], np.array([-300.0, 0.0, 300.0]), C)
    assert sol.valid.tolist() == [False, True, True]  # emission at ~-420 s is outside the data
    assert sol.max_residual < 1e-9
