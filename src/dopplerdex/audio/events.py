"""Doppler event detection and parameter estimation.

detect_events(audio) ->
  feature map (whitened, log-f, harmonic-sum z)  ->  linked ridges  ->  Viterbi refinement
  ->  Doppler-likeness filter  ->  harmonic de-duplication  ->  constant-velocity fit with
  uncertainty  ->  DetectedEvent list.

The fit uses the closed-form straight-pass model (physics.doppler.straight_pass_frequency):
  parameters  f0, v, r_cpa, tau (time the source is at CPA); heard CPA = tau + r_cpa / c.
Residuals are in log-frequency, so the fit is invariant to which harmonic was tracked: the
same (v, r_cpa, tau) explain every harmonic, and only f0 changes by the harmonic number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from dopplerdex.audio.features import FeatureConfig, LogFreqMap, tonal_feature_map
from dopplerdex.audio.tracking import Ridge, link_peaks, stitch_ridges, viterbi_refine
from dopplerdex.physics.atmosphere import C_STANDARD
from dopplerdex.physics.doppler import straight_pass_frequency

F64 = NDArray[np.float64]
PARAM_NAMES = ("f0", "v", "r_cpa", "tau")


@dataclass(frozen=True)
class DopplerFit:
    f0: float
    v: float
    r_cpa: float
    tau: float  # source-at-CPA time (s, recording clock)
    c: float
    cov: F64 = field(repr=False)  # 4x4 covariance of (f0, v, r_cpa, tau), already inflated
    rms_cents: float = 0.0  # residual RMS (cents)
    n_points: int = 0
    success: bool = True

    @property
    def t_cpa_heard(self) -> float:
        return self.tau + self.r_cpa / self.c

    def sigma(self) -> dict[str, float]:
        s = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        d = dict(zip(PARAM_NAMES, s.tolist(), strict=True))
        d["t_cpa_heard"] = float(
            np.sqrt(max(self.cov[3, 3] + self.cov[2, 2] / self.c**2 + 2 * self.cov[2, 3] / self.c, 0))
        )
        return d

    def constraint_gaussian(self) -> tuple[F64, F64]:
        """Mean and 3x3 covariance of (v, r_cpa, t_cpa_heard), with t_heard = tau + r / c."""
        T = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0 / self.c, 1.0]])
        return np.array([self.v, self.r_cpa, self.t_cpa_heard]), T @ self.cov @ T.T

    def conditional_t_sigma(self, wind_sigma: float = 10.0) -> float:
        """Heard-time uncertainty once the search supplies the aircraft's speed (ADS-B, +- wind) and range."""
        _, C = self.constraint_gaussian()
        S = np.array([[C[0, 0] + wind_sigma**2, C[0, 1]], [C[0, 1], C[1, 1]]])
        st = C[2, :2]
        try:
            return float(np.sqrt(max(C[2, 2] - st @ np.linalg.solve(S, st), 0.0)))
        except np.linalg.LinAlgError:
            return float("inf")

    def curve(self, t: F64) -> F64:
        return straight_pass_frequency(t, self.f0, self.v, self.r_cpa, self.tau, self.c)


@dataclass
class DetectedEvent:
    ridge: Ridge
    fit: DopplerFit
    drop_octaves: float  # observed total frequency drop along the ridge
    companions: list[Ridge] = field(default_factory=list)  # other harmonics / orders of the same source
    tier: str = "A"  # "A": speed/range well constrained; "B": timing usable, weakly constrained; "G": ground traffic

    @property
    def harmonic_hint(self) -> int:
        """1 if no companion ridge sits below this one (likely the comb's lowest visible line)."""
        f = np.median(self.ridge.f)
        return int(all(np.median(c.f) > f for c in self.companions))

    @property
    def evidence(self) -> float:
        return float(sum(np.sum(np.clip(r.score, 0, None)) for r in [self.ridge, *self.companions]))

    @property
    def t_start(self) -> float:
        return float(self.ridge.t[0])

    @property
    def t_end(self) -> float:
        return float(self.ridge.t[-1])


@dataclass(frozen=True)
class DetectConfig:
    features: FeatureConfig = FeatureConfig()
    peak_thr: float = 3.5
    min_duration_s: float = 15.0
    min_mean_score: float = 3.5
    min_drop_octaves: float = 0.04  # ~2.8 % total Doppler drop (v ~ 8 m/s); rejects steady tones
    max_rise_fraction: float = 0.4  # straight passes only descend; allow some manoeuvre wiggle
    shape_smooth_s: float = 3.0
    keep_tier_b: bool = True
    plausible_v: tuple[float, float] = (15.0, 180.0)  # m/s: audible tonal sources (props, helicopters, low jets)
    plausible_r: tuple[float, float] = (80.0, 12_000.0)  # m: CPA slant range of an audible pass
    ground_r_max: float = 400.0  # closer than this AND slower than ground_v_max -> road/rail traffic (tier G)
    ground_v_max: float = 45.0
    tier_b_max_t_sigma: float = 12.0  # conditional on speed+range (what the search uses)
    tier_b_min_drop: float = 0.1  # octaves: must show a substantial Doppler drop
    tier_b_min_score: float = 5.0
    min_fragment_s: float = 5.0  # ridges shorter than this are not considered for stitching
    min_fragment_score: float = 3.0
    stitch_gap_s: float = 10.0
    stitch_tol_oct: float = 0.03
    c: float = C_STANDARD
    cov_inflation: float = (
        1.3  # extra model-misspecification margin on top of the autocorrelation VIF (bench-calibrated)
    )
    max_rms_cents: float = 40.0
    max_rel_sigma_v: float = 0.35
    max_rel_sigma_r: float = 0.8
    min_speed: float = 15.0
    explain_tol_oct: float = 0.012  # ridge belongs to an event if its log-f offset from the model is this flat
    merge_sigma: float = 3.0
    single_max_rms_cents: float = 20.0  # stricter gates for events seen on one ridge only
    single_min_duration_s: float = 25.0
    dependent_ratio: float = 5.0
    dependent_overlap: float = 0.8  # fraction of the lone ridge's duration inside the stronger event


# ---------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------


def _model_logf(theta: F64, t: F64, c: float) -> F64:
    lf0, v, lr, tau = theta
    return np.log(straight_pass_frequency(t, np.exp(lf0), v, np.exp(lr), tau, c))


def _variance_inflation(resids: list[F64], max_lag: int = 200) -> float:
    """Integrated autocorrelation time of residual series (Sokal windowing), pooled over segments."""
    taus, ns = [], []
    for r in resids:
        if r.size < 20:
            continue
        x = r - r.mean()
        n = x.size
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[: min(n, max_lag)]
        ac = ac / (ac[0] + 1e-30)
        tau = 1.0
        for k in range(1, ac.size):
            tau += 2 * ac[k]
            if k >= 5 * tau:  # Sokal's automatic window
                break
        taus.append(max(tau, 1.0))
        ns.append(n)
    return float(np.average(taus, weights=ns)) if taus else 1.0


def _initial_guess(t: F64, lf: F64, c: float) -> tuple[float, float, float, float]:
    """Model-free start: plateaus -> (f0, v); steepest descent -> heard CPA; slope there -> r_cpa."""
    n = t.size
    q = max(3, n // 8)
    fa, fr = np.exp(np.median(lf[:q])), np.exp(np.median(lf[-q:]))
    fa = max(fa, fr * 1.0005)
    v0 = float(np.clip(c * (fa - fr) / (fa + fr), 5.0, 0.8 * c))
    f00 = 2 * fa * fr / (fa + fr)
    sm = np.convolve(lf, np.ones(9) / 9, mode="same")
    slope = np.gradient(sm, t)
    k = int(np.argmin(slope[q:-q])) + q if n > 2 * q + 2 else n // 2
    s = max(-slope[k] * f00, 1e-3)  # Hz/s at heard CPA ~ f0 v^2 / (c r)
    r0 = float(np.clip(f00 * v0**2 / (c * s), 50.0, 20000.0))
    return float(f00), v0, r0, float(t[k])


def fit_doppler_joint(
    segments: list[tuple[F64, F64, F64]],
    c: float = C_STANDARD,
    inflation: float = 3.0,
    v_bounds: tuple[float, float] | None = None,
    r_bounds: tuple[float, float] | None = None,
) -> DopplerFit:
    """Joint robust fit of several ridges of ONE source: shared (v, r_cpa, tau), one f0 per ridge.

    segments: list of (t, f, weight). The reported f0 is that of the first segment.
    Residuals are log-frequency, weighted by sqrt(weight); soft-L1 loss tolerates ridge glitches.
    """
    ts = [np.asarray(a, dtype=np.float64) for a, _, _ in segments]
    lfs = [np.log(np.asarray(b, dtype=np.float64)) for _, b, _ in segments]
    ws = [np.sqrt(np.clip(np.asarray(w, dtype=np.float64), 0.1, None)) for _, _, w in segments]
    m = len(segments)
    main = int(np.argmax([x.size for x in ts]))
    f00, v0, r_g, t_h = _initial_guess(ts[main], lfs[main], c)
    # per-ridge f0 starts: median log-f offset of each ridge relative to the main ridge's plateau mean
    lf0s = [np.log(f00) + (np.median(lf) - np.median(lfs[main])) for lf in lfs]
    t_all = np.concatenate(ts)

    def resid(th: F64) -> F64:
        v, lr, tau = th[m:]
        return np.concatenate(
            [
                w * (_model_logf(np.array([th[i], v, lr, tau]), t, c) - lf)
                for i, (t, lf, w) in enumerate(zip(ts, lfs, ws, strict=True))
            ]
        )

    vb = v_bounds or (3.0, 0.95 * c)
    rb = r_bounds or (20.0, 40000.0)
    lo = [x - 1.0 for x in lf0s] + [vb[0], np.log(rb[0]), t_all.min() - 600]
    hi = [x + 1.0 for x in lf0s] + [vb[1], np.log(rb[1]), t_all.max() + 600]
    v0 = float(np.clip(v0, vb[0] * 1.01, vb[1] * 0.99))
    r_g = float(np.clip(r_g, rb[0] * 1.01, rb[1] * 0.99))
    best = None
    for r0 in (r_g, r_g * 0.4, r_g * 2.5, 800.0, 3000.0):
        for vv in (v0, min(v0 * 1.5, 0.9 * c)):
            x0 = np.array([*lf0s, vv, np.log(np.clip(r0, 25.0, 35000.0)), t_h - r0 / c])
            x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
            res = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.004,
                                x_scale=[0.01] * m + [5.0, 0.2, 2.0])  # fmt: skip
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    th = best.x
    r_w = resid(th)
    n = r_w.size
    s2 = float(np.sum(r_w**2) / max(n - (m + 3), 1))
    # Residuals of neighbouring STFT frames are correlated (window overlap, smoothing, slow model
    # misfit); scale the covariance by the integrated autocorrelation time of the residuals.
    vif = _variance_inflation(
        [
            w * (_model_logf(np.array([th[i], *th[m:]]), t, c) - lf)
            for i, (t, lf, w) in enumerate(zip(ts, lfs, ws, strict=True))
        ]
    )
    J = best.jac
    try:
        cov_full = np.linalg.inv(J.T @ J) * s2 * vif * inflation**2
    except np.linalg.LinAlgError:
        cov_full = np.full((m + 3, m + 3), np.inf)
    idx = [0, m, m + 1, m + 2]  # (log f0 of segment 0, v, log r, tau)
    cov_th = cov_full[np.ix_(idx, idx)]
    D = np.diag([np.exp(th[0]), 1.0, np.exp(th[m + 1]), 1.0])
    raw = np.concatenate(
        [_model_logf(np.array([th[i], *th[m:]]), t, c) - lf for i, (t, lf) in enumerate(zip(ts, lfs, strict=True))]
    )
    return DopplerFit(
        f0=float(np.exp(th[0])),
        v=float(th[m]),
        r_cpa=float(np.exp(th[m + 1])),
        tau=float(th[m + 2]),
        c=c,
        cov=D @ cov_th @ D,
        rms_cents=float(1200 / np.log(2) * np.sqrt(np.median(raw**2))),
        n_points=n,
        success=bool(best.success),
    )


def fit_doppler(t: F64, f: F64, w: F64 | None = None, c: float = C_STANDARD, inflation: float = 3.0) -> DopplerFit:
    """Robust weighted least-squares fit of the straight-pass Doppler model to one frequency track."""
    t = np.asarray(t, dtype=np.float64)
    w = np.ones_like(t) if w is None else np.asarray(w, dtype=np.float64)
    return fit_doppler_joint([(t, np.asarray(f, dtype=np.float64), w)], c, inflation)


# ---------------------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------------------


def _doppler_like(r: Ridge, cfg: DetectConfig) -> tuple[bool, float]:
    lf = np.log2(r.f)
    dt = float(np.median(np.diff(r.t))) if r.t.size > 1 else 1.0
    k = max(3, int(round(cfg.shape_smooth_s / dt)))  # judge the shape, not frame-to-frame ridge jitter
    sm = np.convolve(lf, np.ones(k) / k, mode="valid") if lf.size > 2 * k else lf
    drop = float(sm[: max(1, sm.size // 6)].mean() - sm[-max(1, sm.size // 6) :].mean())
    d = np.diff(sm)
    rise = float(np.sum(d[d > 0]) / (np.sum(np.abs(d)) + 1e-12))
    return (drop >= cfg.min_drop_octaves and rise <= cfg.max_rise_fraction), drop


def _explained_by(r: Ridge, ev: DetectedEvent, tol_oct: float, margin_s: float = 30.0) -> bool:
    """True if ridge r is another harmonic/order of event ev: overlapping in time and a constant
    log-frequency offset from ev's fitted Doppler model (any frequency ratio)."""
    if r.t[-1] < ev.t_start - margin_s or r.t[0] > ev.t_end + margin_s:
        return False
    d = np.log2(r.f) - np.log2(ev.fit.curve(r.t))
    med = np.median(d)
    return bool(np.median(np.abs(d - med)) * 1.4826 < tol_oct)


def _merge_consistent(events: list[DetectedEvent], cfg: DetectConfig) -> list[DetectedEvent]:
    """Merge events that overlap in time and whose independent fits agree (same aircraft seen on
    different harmonic orders). Agreement: heard-CPA and speed within `merge_sigma` combined sigmas
    (plus small absolute floors)."""
    events = sorted(events, key=lambda e: -e.evidence)
    out: list[DetectedEvent] = []
    for e in events:
        host = None
        for h in out:
            if min(e.t_end, h.t_end) - max(e.t_start, h.t_start) < 0.25 * min(e.ridge.duration, h.ridge.duration):
                continue
            se, sh = e.fit.sigma(), h.fit.sigma()
            dt_ok = (
                abs(e.fit.t_cpa_heard - h.fit.t_cpa_heard)
                <= cfg.merge_sigma * np.hypot(se["t_cpa_heard"], sh["t_cpa_heard"]) + 1.5
            )
            dv_ok = abs(e.fit.v - h.fit.v) <= cfg.merge_sigma * np.hypot(se["v"], sh["v"]) + 0.05 * h.fit.v
            if dt_ok and dv_ok:
                host = h
                break
        if host is None:
            out.append(e)
        else:
            host.companions.extend([e.ridge, *e.companions])
    return out


def _is_dependent(e: DetectedEvent, events: list[DetectedEvent], cfg: DetectConfig) -> bool:
    """A lone ridge lying inside a much stronger event's time span is most likely an unmerged
    fragment of that aircraft (another engine order distorted by ground interference)."""
    if e.companions:
        return False
    for h in events:
        if h is e or h.evidence < cfg.dependent_ratio * e.evidence:
            continue
        overlap = min(e.t_end, h.t_end) - max(e.t_start, h.t_start)
        if overlap >= cfg.dependent_overlap * e.ridge.duration:
            return True
    return False


def _plausible(fit: DopplerFit, cfg: DetectConfig) -> bool:
    return cfg.plausible_v[0] <= fit.v <= cfg.plausible_v[1] and cfg.plausible_r[0] <= fit.r_cpa <= cfg.plausible_r[1]


def classify_ground(events: list[DetectedEvent], cfg: DetectConfig) -> None:
    """Mark road/rail traffic as tier G (in place).

    Road vehicles pass within a few hundred metres at highway speed; aircraft CPAs are typically at
    least their altitude (~km). Tier-G events stay visible in reports but are never used to search.
    """
    for e in events:
        if e.fit.r_cpa < cfg.ground_r_max and e.fit.v < cfg.ground_v_max:
            e.tier = "G"


def _passes_quality(fit: DopplerFit, cfg: DetectConfig) -> bool:
    s = fit.sigma()
    return (
        fit.success
        and fit.rms_cents <= cfg.max_rms_cents
        and fit.v >= cfg.min_speed
        and s["v"] <= cfg.max_rel_sigma_v * fit.v
        and s["r_cpa"] <= cfg.max_rel_sigma_r * fit.r_cpa
    )


def detect_events(
    x: F64, fs: float, cfg: DetectConfig = DetectConfig(), t0: float = 0.0, diagnostics: list | None = None
) -> tuple[list[DetectedEvent], LogFreqMap]:
    """Detect Doppler events. If `diagnostics` is a list, (stage, reason, ridge, fit) tuples for every
    rejected candidate are appended to it."""

    def reject(stage: str, reason: str, ridge: Ridge | None, fit: DopplerFit | None = None) -> None:
        if diagnostics is not None:
            diagnostics.append((stage, reason, ridge, fit))

    _, hs = tonal_feature_map(x, fs, cfg.features, t0)
    tracks = link_peaks(hs, thr=cfg.peak_thr)
    frag_frames = int(cfg.min_fragment_s / hs.dt)
    fragments = [
        viterbi_refine(hs, tr)
        for tr in tracks
        if (tr.frames[-1] - tr.frames[0]) >= frag_frames and len(tr.frames) >= 0.5 * frag_frames
    ]
    fragments = [r for r in fragments if r.mean_score >= cfg.min_fragment_score]
    ridges: list[Ridge] = []
    for r in stitch_ridges(fragments, cfg.stitch_gap_s, cfg.stitch_tol_oct):
        if r.duration < cfg.min_duration_s:
            continue
        if r.mean_score < cfg.min_mean_score:
            reject("ridge", f"mean score {r.mean_score:.1f} < {cfg.min_mean_score}", r)
            continue
        ok, drop = _doppler_like(r, cfg)
        if ok:
            ridges.append(r)
        else:
            reject("ridge", f"not Doppler-like (drop {drop:+.3f} oct)", r)
    ridges.sort(key=lambda r: -float(np.sum(np.clip(r.score, 0, None))))
    events: list[DetectedEvent] = []
    for r in ridges:
        host = next((e for e in events if _explained_by(r, e, cfg.explain_tol_oct)), None)
        if host is not None:
            host.companions.append(r)
            continue
        fit = fit_doppler(r.t, r.f, np.clip(r.score, 0.5, 20.0), cfg.c, cfg.cov_inflation)
        _, drop = _doppler_like(r, cfg)
        if not _passes_quality(fit, cfg):
            if (
                cfg.keep_tier_b
                and fit.rms_cents <= cfg.max_rms_cents
                and fit.conditional_t_sigma() <= cfg.tier_b_max_t_sigma
                and drop >= cfg.tier_b_min_drop
                and r.mean_score >= cfg.tier_b_min_score
            ):
                events.append(DetectedEvent(r, fit, drop, tier="B"))
                continue
            sg = fit.sigma()
            reject(
                "fit",
                f"quality: rms {fit.rms_cents:.0f}c v {fit.v:.0f}+-{sg['v']:.0f} r {fit.r_cpa:.0f}+-{sg['r_cpa']:.0f}",
                r,
                fit,
            )
            continue
        events.append(DetectedEvent(r, fit, drop))
    events = _merge_consistent(events, cfg)
    for e in events:
        if e.companions:
            segs = [(x.t, x.f, np.clip(x.score, 0.5, 20.0)) for x in [e.ridge, *e.companions]]
            e.fit = fit_doppler_joint(segs, cfg.c, cfg.cov_inflation)
            e.tier = "A" if _passes_quality(e.fit, cfg) else "B"
    kept = []
    for e in events:
        if e.companions or (
            e.fit.rms_cents <= cfg.single_max_rms_cents and e.ridge.duration >= cfg.single_min_duration_s
        ):
            kept.append(e)
        else:
            reject("event", f"single ridge: rms {e.fit.rms_cents:.0f}c dur {e.ridge.duration:.0f}s", e.ridge, e.fit)
    events = []
    for e in kept:
        if _is_dependent(e, kept, cfg):
            reject("event", "dependent fragment", e.ridge, e.fit)
        else:
            events.append(e)
    plausible = []
    for e in events:
        if _plausible(e.fit, cfg):
            plausible.append(e)
            continue
        # The unconstrained optimum is unphysical, but the curve (and its timing) may still be real:
        # refit inside physically plausible bounds and keep it as a timing-quality (tier B) event.
        segs = [(x.t, x.f, np.clip(x.score, 0.5, 20.0)) for x in [e.ridge, *e.companions]]
        rf = fit_doppler_joint(segs, cfg.c, cfg.cov_inflation, v_bounds=cfg.plausible_v, r_bounds=cfg.plausible_r)
        if rf.rms_cents <= cfg.max_rms_cents and rf.conditional_t_sigma() <= cfg.tier_b_max_t_sigma:
            e.fit, e.tier = rf, "B"
            plausible.append(e)
        else:
            reject(
                "event", f"implausible fit v={e.fit.v:.0f} r={e.fit.r_cpa:.0f} (bounded refit failed)", e.ridge, e.fit
            )
    classify_ground(plausible, cfg)
    events = sorted(plausible, key=lambda e: e.fit.t_cpa_heard)
    return events, hs
