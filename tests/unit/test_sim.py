"""Simulator correctness: rendered audio must obey the physics it claims to model."""

import numpy as np
import pytest
from scipy.signal import welch

from dopplerdex.physics.doppler import straight_pass_frequency
from dopplerdex.sim.noise import AmbientConfig, NoiseProfile, render_ambient, texture_noise
from dopplerdex.sim.render import Atmosphere, Device, Ground
from dopplerdex.sim.scenario import Scenario, simulate, straight_pass
from dopplerdex.sim.sources import PRESETS, SourceModel, Tone, make_shaft_phase

FS = 8000.0


def _pure_tone_source(f0=100.0):
    return SourceModel("tone", "piston", f0, (Tone(1.0, 120.0),), None, rpm_wander=0.0)


def _quiet(**kw):
    return Scenario(
        ambient=None,
        ground=Ground(enabled=False),
        atmosphere=Atmosphere(turbulence_db=0.0),
        device=Device(self_noise_db=-50, highpass_hz=None, bits=24),
        fs=FS,
        **kw,
    )


def test_shaft_phase_derivative_matches_rate():
    rng = np.random.default_rng(0)
    sp = make_shaft_phase(PRESETS["C172"], 0, 60, rng)
    t = np.linspace(1, 59, 5001)
    inst = np.gradient(sp(t), t) / (2 * np.pi)
    np.testing.assert_allclose(inst, np.interp(t, sp.t, sp.f_shaft), rtol=1e-3)


def test_rendered_tone_follows_closed_form_doppler():
    """Instantaneous frequency of the rendered pressure == analytic received frequency."""
    v, r, alt, tc, f0 = 70.0, 900.0, 500.0, 30.0, 100.0
    ap = straight_pass(_pure_tone_source(f0), tc, r, alt, v, 0.0)
    res = simulate(_quiet(aircraft=[ap], duration=60.0))
    x = res.components["aircraft"]
    from scipy.signal import hilbert

    ph = np.unwrap(np.angle(hilbert(x)))
    t = np.arange(x.size) / FS
    inst = np.gradient(ph, t) / (2 * np.pi)
    sel = (t > 10) & (t < 50)
    c = res.scenario.atmosphere.c
    expect = straight_pass_frequency(t[sel], f0, v, r, tc, c)
    # median-filter-free check on a decimated grid (Hilbert IF is noisy at sample level)
    k = 400
    got = inst[sel][: (sel.sum() // k) * k].reshape(-1, k).mean(axis=1)
    exp = expect[: (sel.sum() // k) * k].reshape(-1, k).mean(axis=1)
    np.testing.assert_allclose(got, exp, rtol=2e-4)


def test_spherical_spreading_level():
    """Received tone level == source level - 20 log10(r) (no ground, absorption tiny at 100 Hz)."""
    r = 1000.0
    ap = straight_pass(_pure_tone_source(100.0), 30.0, r, r, 0.01, 0.0)  # (almost) hovering overhead
    res = simulate(_quiet(aircraft=[ap], duration=40.0))
    x = res.components["aircraft"][int(10 * FS) : int(35 * FS)]
    rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) / 20e-6)
    assert rms_db == pytest.approx(120.0 - 60.0, abs=0.3)


def test_tone_frequencies_present_in_rendered_preset():
    ap = straight_pass("C172", 30.0, 1200.0, 600.0, 55.0, 70.0)
    res = simulate(Scenario([ap], duration=60, fs=FS, ambient=None, seed=3))
    x = res.components["aircraft"][int(25 * FS) : int(40 * FS)]
    f, P = welch(x, FS, nperseg=8192)
    tr = res.truth[0]
    i = np.argmin(np.abs(tr.control_t - 32.0))
    for k in range(3):  # first three BPF harmonics must stand well above the broadband
        fk = tr.tone_freqs[k, i]
        band = (f > fk - 8) & (f < fk + 8)
        ref = (f > fk + 25) & (f < fk + 60)
        assert P[band].max() > 30 * np.median(P[ref])


def test_ground_reflection_creates_interference_nulls():
    """Hovering source overhead, rigid ground, mic at h: broadband nulls at (2n+1) c / (4 h)."""
    from dopplerdex.sim.sources import Broadband

    src = SourceModel("bb", "jet", 1.0, (), Broadband(130.0, 500.0, 0.0, 0.0), rpm_wander=0.0)
    ap = straight_pass(src, 20.0, 300.0, 300.0, 0.01, 0.0)
    h = 2.0
    scn = _quiet(aircraft=[ap], duration=40.0)
    scn.ground = Ground(mic_height=h, flow_resistivity=None, enabled=True)
    res = simulate(scn)
    x = res.components["aircraft"][int(8 * FS) : int(35 * FS)]
    f, P = welch(x, FS, nperseg=4096)
    c = scn.atmosphere.c
    for n in range(3):
        fn = (2 * n + 1) * c / (4 * h)
        near = (f > fn - 4) & (f < fn + 4)
        peak = (f > fn + c / (4 * h) - 6) & (f < fn + c / (4 * h) + 6)  # constructive in between
        assert P[near].min() < 0.05 * P[peak].max()


def test_ambient_components_and_enf():
    rng = np.random.default_rng(0)
    parts = render_ambient(AmbientConfig(), 4000.0, 30.0, rng)
    for k in ("background", "wind", "mains", "tones", "events"):
        assert parts[k].shape == (120000,)
    assert np.all(np.abs(parts["_enf_hz"] - 60.0) <= 0.05)


def test_texture_noise_matches_band_statistics():
    rng = np.random.default_rng(1)
    fs = 8000.0
    # reference: brown-ish noise with a slow amplitude modulation
    n = int(60 * fs)
    ref = np.cumsum(rng.normal(size=n))
    ref -= np.convolve(ref, np.ones(4001) / 4001, mode="same")
    ref *= 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / fs / 10)
    prof = NoiseProfile.fit_band_levels(ref, fs)
    syn = texture_noise(prof, fs, 60.0, rng)
    prof2 = NoiseProfile.fit_band_levels(syn, fs)
    m1, m2 = prof.levels_db.mean(axis=1), prof2.levels_db.mean(axis=1)
    k = min(m1.size, m2.size)
    assert np.median(np.abs(m1[:k] - m2[:k])) < 2.0  # spectral shape within 2 dB
    s1, s2 = prof.levels_db.std(axis=1), prof2.levels_db.std(axis=1)
    assert abs(np.median(s1[:k]) - np.median(s2[:k])) < 2.0  # level variability similar


@pytest.mark.parametrize("profile", ["phone_nb", "mp3_64k"])
def test_codec_roundtrip_keeps_length_alignment_and_tone(profile):
    from dopplerdex.sim.codec import codec_roundtrip

    fs = 16000
    t = np.arange(int(5 * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 440 * t) * (t > 1)  # onset at 1 s for alignment
    y = codec_roundtrip(x, fs, profile)
    assert y.size == x.size
    on = np.argmax(np.abs(y) > 0.05) / fs
    assert abs(on - 1.0) < 0.03
    f, P = welch(y[fs * 2 :], fs, nperseg=4096)
    assert abs(f[np.argmax(P)] - 440) < 5


def test_absorption_uses_range_at_the_right_time():
    """A pass far from the window start must not be attenuated with the window-start range."""
    from dopplerdex.sim.render import Atmosphere, apply_absorption

    fs = 8000.0
    t_start = 1.79e9  # absolute clock, as with real ADS-B tracks
    n = int(60 * fs)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    t = t_start + np.array([-10.0, 0.0, 30.0, 70.0])
    r = np.array([20000.0, 20000.0, 1000.0, 1000.0])  # far at the start, close for the second half
    y = apply_absorption(x, fs, t, r, Atmosphere(), t_start=t_start)
    y_near = apply_absorption(x, fs, t, np.full(4, 1000.0), Atmosphere(), t_start=t_start)
    y_far = apply_absorption(x, fs, t, np.full(4, 20000.0), Atmosphere(), t_start=t_start)
    late = slice(int(40 * fs), int(55 * fs))

    def p(z):
        return 10 * np.log10(np.mean(z[late] ** 2))

    assert abs(p(y) - p(y_near)) < 0.5  # second half attenuated like 1 km ...
    assert p(y) - p(y_far) > 10.0  # ... and not like the 20 km window-start range
