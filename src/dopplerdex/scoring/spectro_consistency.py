"""Spectrogram consistency of a (place, T0) hypothesis with every predicted tonal pass.

For each predicted propeller/helicopter pass, the log-frequency feature map is Doppler-compensated
frame by frame using the predicted D(t): column j is read at bin b0 + bpo*log2(D_j) (+ harmonics).
If the hypothesis is right, the pass becomes a horizontal line at its (unknown) f0 bin b0, so the
time-summed profile over b0 has a sharp peak. The peak's robust z-score is the pass's *evidence*.

Evidence = robust z of max_b [compensated(b) - uncompensated(b)] (steady tones cancel).
Score = sum over predicted passes of w(r_cpa) * (evidence - threshold): confirmed passes add, close
passes that should be obvious but are absent subtract (soft silence filter). Aircraft without ADS-B
can only add unexplained curves, which this score never penalises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dopplerdex.audio.features import LogFreqMap
from dopplerdex.search.oracle import PredictedPass

TONAL = ("piston", "turboprop", "helicopter", "unknown")


@dataclass(frozen=True)
class PassEvidence:
    icao: str
    aircraft_class: str
    r_cpa: float
    t_heard_rec: float  # heard CPA on the recording clock
    evidence: float  # robust z of the Doppler-compensated profile peak
    f0: float  # Hz of the best profile bin (lowest visible line)
    weight: float


def pass_evidence(fm: LogFreqMap, t0: float, p: PredictedPass, window_s: float = 75.0, n_harm: int = 3,
                  f_lo: float = 45.0, f_hi: float = 650.0) -> tuple[float, float] | None:  # fmt: skip
    """Evidence (robust z of the compensated-profile peak) and f0 for one predicted pass."""
    t_abs = fm.t + t0
    j = np.flatnonzero(np.abs(t_abs - p.t_cpa_heard) <= window_s)
    if j.size < 40:
        return None
    ok = np.interp(t_abs[j], p.t_r, p.valid.astype(float)) > 0.5
    j = j[ok]
    if j.size < 40:
        return None
    logd = np.interp(t_abs[j], p.t_r, np.log(p.doppler))
    shift = fm.bpo * logd / np.log(2.0)  # bins
    b_lo = int(np.floor(np.log2(f_lo / fm.f[0]) * fm.bpo))
    b_hi = int(np.ceil(np.log2(f_hi / fm.f[0]) * fm.bpo))
    b0 = np.arange(max(b_lo, 0), min(b_hi, fm.z.shape[0]))
    if b0.size < 20:
        return None

    def profile(sh: np.ndarray) -> np.ndarray:
        prof = np.zeros(b0.size)
        for h in range(1, n_harm + 1):
            bins = np.rint(b0[:, None] + sh[None, :] + fm.bpo * np.log2(h)).astype(int)  # (B0, J)
            valid = (bins >= 0) & (bins < fm.z.shape[0])
            vals = np.where(valid, fm.z[np.clip(bins, 0, fm.z.shape[0] - 1), j[None, :]], 0.0)
            prof += vals.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        return prof / n_harm

    comp = profile(shift)  # Doppler-compensated: the predicted pass becomes a horizontal line
    flat = profile(np.full_like(shift, np.median(shift)))  # uncompensated: steady tones line up here
    # Contrast: a real pass lines up ONLY under its own Doppler curve; steady tones line up either way.
    diff = comp - flat
    med = np.median(diff)
    mad = 1.4826 * np.median(np.abs(diff - med)) + 1e-9
    k = int(np.argmax(diff))
    return float((diff[k] - med) / mad), float(fm.f[0] * 2 ** (b0[k] / fm.bpo))


def detectability_weight(r_cpa: float, r_full: float = 2500.0, r_zero: float = 7000.0) -> float:
    """1 for close passes (should certainly be visible), fading to 0 by r_zero."""
    if r_cpa <= r_full:
        return 1.0
    return float(max(0.0, (r_zero - r_cpa) / (r_zero - r_full)))


def spectro_consistency(fm: LogFreqMap, t0: float, passes: list[PredictedPass], duration: float,
                        threshold: float = 4.0, r_max: float = 7000.0) -> tuple[float, list[PassEvidence]]:  # fmt: skip
    total = 0.0
    out: list[PassEvidence] = []
    for p in passes:
        if p.flight.aircraft_class not in TONAL or p.r_cpa > r_max:
            continue
        t_rec = p.t_cpa_heard - t0
        if t_rec < 30 or t_rec > duration - 30:
            continue
        res = pass_evidence(fm, t0, p)
        if res is None:
            continue
        e, f0 = res
        w = detectability_weight(p.r_cpa)
        total += w * (min(e, 15.0) - threshold)
        out.append(PassEvidence(p.flight.icao, p.flight.aircraft_class, p.r_cpa, t_rec, e, f0, w))
    return total, out
