"""Ground-reflection (Lloyd's mirror) events: the multipath delay curve of a passing aircraft.

A microphone at height h above reflecting ground hears the direct sound and a copy reflected by the
ground. For a source at elevation angle theta and range >> h the extra path is ~2 h sin(theta), so the
two copies are delayed by

    tau(t) = 2 h sin(theta(t)) / c.

The received spectrum is multiplied by |1 + R exp(-i 2 pi f tau)|^2: the log spectrum ripples with period
1/tau, and the power cepstrum peaks at quefrency tau. This works for broadband sources (jets, most close
passes) that have no usable tonal line. Method after Lo & Ferguson (IEEE TAES 2002/2003).

For a straight level pass, sin(theta) = H / sqrt(S0^2 + v^2 (t - t_c)^2), hence

    tau(t) = tau_max / sqrt(1 + ((t - t_c) / T)^2),   T = S0 / v.

One microphone of unknown height gives t_c, T and tau_max; against a real ADS-B track the whole curve
h sin(theta(t)) is predicted up to the scale h, which the scorer fits.

Pipeline: cepstrogram (log spectrum detrended, projected on cosines, per-frame robust z; steady
quefrencies such as fixed reflections or tone rahmonics removed with a running temporal median) ->
Viterbi track with an off state -> least-squares pass fit -> physical gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

F64 = NDArray[np.float64]


@dataclass
class MultipathConfig:
    frame_s: float = 0.5
    hop_s: float = 0.25
    f_lo: float = 150.0
    f_hi: float = 2000.0
    q_lo: float = 1.5e-3  # s; below this the cepstrum is dominated by the smooth source spectrum
    q_hi: float = 40e-3  # s; mic up to ~7 m (2 h / c)
    dq: float = 0.05e-3
    steady_s: float = 120.0  # running median over time removes constant-delay structure
    z_on: float = 4.0  # per-frame robust z for the "on" state
    birth: float = 8.0  # Viterbi cost to start a track
    max_jump_q: float = 1.2e-3  # s per hop
    min_span_s: float = 10.0  # noise / highway negatives: tracks <= 10 s; real passes >= 79 s (synthetic)
    min_median_z: float = 10.0  # negatives <= 8.8; weakest synthetic pass 13
    min_contrast: float = 1.15  # fitted tau_max / tau at the track ends
    max_rel_rms: float = 0.05  # weighted RMS residual / tau_max
    t_range: tuple[float, float] = (0.8, 150.0)  # T = S0 / v
    max_tracks: int = 12


@dataclass
class MultipathEvent:
    t: F64 = field(repr=False)  # track times (recording s)
    tau: F64 = field(repr=False)  # delay (s)
    w: F64 = field(repr=False)  # weights (cepstral z)
    t_cpa: float = 0.0  # heard time of maximum elevation
    T: float = 0.0  # S0 / v (s)
    tau_max: float = 0.0
    rel_rms: float = 0.0
    median_z: float = 0.0
    v: float = float("nan")  # speed from the travel-time skew (weakly determined)
    tier: str = "M"
    kind: str = "multipath"

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])


def cepstrogram(x: F64, fs: float, cfg: MultipathConfig) -> tuple[F64, F64, F64]:
    """(frame times, quefrencies, z-scored cepstral power with steady quefrencies removed)."""
    n = int(round(cfg.frame_s * fs))
    hop = int(round(cfg.hop_s * fs))
    nfft = int(2 ** np.ceil(np.log2(n)))
    win = np.hanning(n)
    n_fr = max(0, 1 + (len(x) - n) // hop)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    fb = f[band]
    q = np.arange(cfg.q_lo, cfg.q_hi, cfg.dq)
    basis = np.cos(2 * np.pi * np.outer(fb, q))  # (n_f, n_q)
    basis -= basis.mean(axis=0)
    # polynomial trend in f (the source spectrum), removed by projection
    u = (fb - fb.mean()) / (np.ptp(fb) / 2)
    P = np.vstack([u**k for k in range(5)]).T
    proj = np.eye(fb.size) - P @ np.linalg.pinv(P)
    B = proj @ basis
    out = np.empty((n_fr, q.size))
    chunk = 2048
    for s0 in range(0, n_fr, chunk):
        idx = np.arange(s0, min(s0 + chunk, n_fr))
        frames = np.lib.stride_tricks.sliding_window_view(x, n)[idx * hop] * win
        spec = np.abs(np.fft.rfft(frames, nfft, axis=1))[:, band] ** 2
        ls = np.log(spec + 1e-12 * spec.mean() + 1e-30)
        out[idx] = ls @ B
    c = out**2 * np.sign(out)  # emphasise positive cepstral peaks
    # steady quefrencies (fixed reflections, rahmonics of steady tones) -> running temporal median
    k = max(3, int(cfg.steady_s / cfg.hop_s) | 1)
    if n_fr > 8:
        c = c - median_filter(c, size=(min(k, n_fr - (1 - n_fr % 2)), 1), mode="nearest")
    med = np.median(c, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(c - med), axis=1, keepdims=True) + 1e-12
    z = (c - med) / mad
    t = (np.arange(n_fr) * hop + n / 2) / fs
    return t, q, z


@nb.njit(cache=True)
def _viterbi_on_off(z, z_on, birth, max_jump):
    """Best path over states {off, q_0..q_{n-1}}; on-state gain z - z_on, jumps <= max_jump bins
    (penalised 0.1 per bin), starting a track costs birth. Returns state per frame (-1 = off)."""
    n_t, n_q = z.shape
    prev = np.zeros(n_q + 1)
    prev[1:] = -1e18
    back = np.zeros((n_t, n_q + 1), np.int32)
    cur = np.empty(n_q + 1)
    for t in range(n_t):
        # off: from off or any on (ending is free)
        best_off = prev[0]
        arg_off = 0
        for j in range(1, n_q + 1):
            if prev[j] > best_off:
                best_off = prev[j]
                arg_off = j
        cur[0] = best_off
        back[t, 0] = arg_off
        for j in range(n_q):
            b = prev[0] - birth
            arg = 0
            lo = max(0, j - max_jump)
            hi = min(n_q, j + max_jump + 1)
            for i in range(lo, hi):
                v = prev[i + 1] - 0.1 * abs(i - j)
                if v > b:
                    b = v
                    arg = i + 1
            cur[j + 1] = b + z[t, j] - z_on
            back[t, j + 1] = arg
        prev[:] = cur
    path = np.empty(n_t, np.int32)
    s = 0
    best = prev[0]
    for j in range(1, n_q + 1):
        if prev[j] > best:
            best = prev[j]
            s = j
    for t in range(n_t - 1, -1, -1):
        path[t] = s - 1
        s = back[t, s]
    return path


def _model(p: F64, t: F64, c: float = 340.0) -> F64:
    """tau at reception times t for a straight level pass, with the sound travel time included:
    t = t_e + S0 sqrt(1 + u^2) / c, u = (t_e - tc_e) / T, S0 = T v (this skews the curve; the skew
    measures v / c)."""
    tau_max, tc_e, T, v = p
    S0 = T * v
    te = t - S0 / c
    for _ in range(8):
        u = (te - tc_e) / T
        sq = np.sqrt(1.0 + u * u)
        te = te - (te + S0 * sq / c - t) / (1.0 + v * u / (c * sq))
    return tau_max / np.sqrt(1.0 + ((te - tc_e) / T) ** 2)


def _fit(t: F64, tau: F64, w: F64, cfg: MultipathConfig) -> tuple[F64, float, float] | None:
    i = int(np.argmax(tau))
    half = tau[i] / np.sqrt(2.0)
    above = np.flatnonzero(tau >= half)
    T0 = max(cfg.t_range[0], 0.5 * (t[above[-1]] - t[above[0]]) if above.size > 1 else 5.0)
    sw = np.sqrt(w / w.mean())

    def res(p):
        return sw * (_model(p, t) - tau) / tau[i]

    lo = [0.5 * tau[i], t[0] - 60.0, cfg.t_range[0], 20.0]
    hi = [2.0 * tau[i], t[-1] + 30.0, cfg.t_range[1], 320.0]
    best = None
    for v0 in (60.0, 150.0):  # speed is weakly determined (curve skew); two starts avoid a poor local fit
        p0 = np.clip(
            [tau[i], t[i] - T0 * v0 / 340.0, min(T0, cfg.t_range[1] * 0.99), v0],
            np.array(lo) + 1e-9,
            np.array(hi) - 1e-9,
        )
        try:
            r = least_squares(res, p0, bounds=(lo, hi))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    r = best
    pred = _model(r.x, t)
    rel = float(np.sqrt(np.average(((pred - tau) / r.x[0]) ** 2, weights=w)))  # residual / tau_max
    # improvement over a constant delay (steady structure is flat in time)
    rss1 = float(np.sum(w * (pred - tau) ** 2))
    rss0 = float(np.sum(w * (tau - np.average(tau, weights=w)) ** 2))
    n = t.size
    F = ((rss0 - rss1) / 2) / max(rss1 / max(n - 3, 1), 1e-30)
    p = float(f_dist.sf(F, 2, max(n - 3, 1)))
    return r.x, rel, p


def detect_multipath(
    x: F64, fs: float, cfg: MultipathConfig | None = None, diag: dict | None = None
) -> list[MultipathEvent]:
    """Ground-reflection delay events in a recording (any length), best first."""
    cfg = cfg or MultipathConfig()
    t, q, z = cepstrogram(np.asarray(x, dtype=np.float64), fs, cfg)
    if t.size < 16:
        return []
    zz = z.copy()
    max_jump = max(1, int(round(cfg.max_jump_q / cfg.dq)))
    min_frames = int(np.ceil(cfg.min_span_s / cfg.hop_s))
    events: list[MultipathEvent] = []
    rejected: list[str] = []
    for _ in range(cfg.max_tracks):
        path = _viterbi_on_off(zz, cfg.z_on, cfg.birth, max_jump)
        on = path >= 0
        if not on.any():
            break
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
        found = False
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            j = path[a:b]
            # mask what this segment explains so the next pass finds other tracks
            for k, jj in zip(range(a, b), j, strict=True):
                zz[k, max(0, jj - 2 * max_jump) : jj + 2 * max_jump + 1] = -1e3
            found = True
            if b - a < min_frames:
                continue
            ts, js = t[a:b], j
            zs = z[np.arange(a, b), js]
            # parabolic sub-bin refinement
            jl, jr = np.clip(js - 1, 0, q.size - 1), np.clip(js + 1, 0, q.size - 1)
            yl, y0, yr = z[np.arange(a, b), jl], zs, z[np.arange(a, b), jr]
            den = yl - 2 * y0 + yr
            off = np.where(np.abs(den) > 1e-9, 0.5 * (yl - yr) / den, 0.0).clip(-0.5, 0.5)
            tau = q[js] + off * cfg.dq
            w = np.clip(zs, 0.5, 30.0)
            mz = float(np.median(zs))
            if mz < cfg.min_median_z:
                rejected.append(f"{ts[0]:.0f}s: median z {mz:.1f}")
                continue
            fit = _fit(ts, tau, w, cfg)
            if fit is None:
                continue
            (tau_max, tc_e, T, v), rel, p = fit
            tc = tc_e + T * v / 340.0  # heard time of the maximum elevation
            contrast = tau_max / max(float(_model(fit[0], np.array([ts[0], ts[-1]])).min()), 1e-9)
            why = None
            if p > 1e-4:
                why = f"flat (p={p:.1e})"
            elif contrast < cfg.min_contrast:
                why = f"contrast {contrast:.2f}"
            elif not ts[0] <= tc <= ts[-1]:
                why = "CPA outside track"
            elif not cfg.t_range[0] <= T <= cfg.t_range[1] * 0.999:
                why = f"T {T:.1f}"
            elif rel > cfg.max_rel_rms:
                why = f"rel rms {rel:.3f}"
            elif v >= 0.98 * 320.0 or v <= 1.02 * 20.0:
                why = f"speed fit at its bound ({v:.0f} m/s)"
            if why:
                rejected.append(f"{ts[0]:.0f}s: {why}")
                continue
            events.append(MultipathEvent(ts, tau, w, float(tc), float(T), float(tau_max), rel, mz, v=float(v)))
        if not found:
            break
    if diag is not None:
        diag["rejected"] = rejected
    events.sort(key=lambda e: -e.median_z * e.t.size)
    return events


def predicted_ratio_rms(ev: MultipathEvent, t_abs: F64, t_r: F64, sin_el: F64, valid: NDArray[np.bool_],
                        h_bounds: tuple[float, float] = (0.2, 30.0), c: float = 343.0) -> float:  # fmt: skip
    """RMS (relative to the peak delay) between the observed delays and the scale-fitted predicted
    2 h sin(theta) / c.

    t_abs: absolute reception times of the event samples (T0 + ev.t). Returns inf if the prediction
    does not cover enough of the track. The scale (mic height) is fitted in closed form and clipped to
    h_bounds."""
    m = valid & np.isfinite(sin_el) & (sin_el > 0)
    if m.sum() < 4:
        return np.inf
    ok = (t_abs >= t_r[m][0]) & (t_abs <= t_r[m][-1])
    if ok.mean() < 0.6:
        return np.inf
    s = np.interp(t_abs[ok], t_r[m], sin_el[m])
    y, w = ev.tau[ok], ev.w[ok]
    k = float(np.sum(w * y * s) / max(np.sum(w * s * s), 1e-30))  # tau ~ k sin(theta), k = 2h/c
    k = float(np.clip(k, 2 * h_bounds[0] / c, 2 * h_bounds[1] / c))
    pred = k * s
    return float(np.sqrt(np.average(((y - pred) / max(float(pred.max()), 1e-6)) ** 2, weights=w)))
