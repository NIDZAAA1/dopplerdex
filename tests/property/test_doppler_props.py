"""Property-based tests: the retarded-time solver agrees with closed-form physics across
randomised geometries, and basic invariants hold."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dopplerdex.physics.doppler import solve_retarded, straight_pass_frequency
from dopplerdex.physics.trajectory import Trajectory

C = 343.0

geom = st.fixed_dictionaries(
    {
        "v": st.floats(20.0, 250.0),
        "r_cpa": st.floats(80.0, 12000.0),
        "elev_frac": st.floats(0.05, 1.0),
        "heading": st.floats(0.0, 360.0),
        "t_cpa": st.floats(-500.0, 500.0),
        "f0": st.floats(20.0, 2000.0),
    }
)


def _traj(g):
    alt = g["r_cpa"] * g["elev_frac"]
    horiz = np.sqrt(max(g["r_cpa"] ** 2 - alt**2, 0.0))
    h = np.radians(g["heading"])
    vel = np.array([g["v"] * np.sin(h), g["v"] * np.cos(h), 0.0])
    side = np.array([np.cos(h), -np.sin(h), 0.0])
    span = 2 * (g["r_cpa"] + 20000) / g["v"]
    return Trajectory.straight(side * horiz + [0, 0, alt], vel, g["t_cpa"], g["t_cpa"] - span, g["t_cpa"] + span, 2.0)


@settings(max_examples=150, deadline=None)
@given(geom)
def test_solver_equals_closed_form(g):
    tr = _traj(g)
    t_r = g["t_cpa"] + np.linspace(-300, 300, 301)
    sol = solve_retarded(tr, [0, 0, 0], t_r, C)
    expect = straight_pass_frequency(t_r, g["f0"], g["v"], g["r_cpa"], g["t_cpa"], C)
    np.testing.assert_allclose(g["f0"] * sol.doppler, expect, rtol=1e-7)
    assert sol.max_residual < 1e-7


@settings(max_examples=100, deadline=None)
@given(geom)
def test_received_frequency_monotone_decreasing_for_straight_pass(g):
    t_r = g["t_cpa"] + np.linspace(-600, 600, 1001)
    f = straight_pass_frequency(t_r, g["f0"], g["v"], g["r_cpa"], g["t_cpa"], C)
    assert np.all(np.diff(f) <= 1e-9)
    # bounded by the approach/recede plateaus
    m = g["v"] / C
    assert f.max() <= g["f0"] / (1 - m) + 1e-9 and f.min() >= g["f0"] / (1 + m) - 1e-9


@settings(max_examples=60, deadline=None)
@given(geom, st.floats(-20, 20), st.floats(-20, 20))
def test_wind_solver_converges(g, wx, wy):
    tr = _traj(g)
    t_r = g["t_cpa"] + np.linspace(-200, 200, 101)
    sol = solve_retarded(tr, [0, 0, 0], t_r, C, wind=[wx, wy, 0.0])
    assert sol.max_residual < 1e-7
    assert sol.iterations <= 8
