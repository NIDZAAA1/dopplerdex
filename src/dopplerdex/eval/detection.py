"""Match detected events to ground truth and compute detection / estimation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dopplerdex.audio.events import DetectedEvent
from dopplerdex.sim.scenario import EventTruth


@dataclass(frozen=True)
class Match:
    truth_label: str
    r_cpa_true: float
    v_true: float
    t_heard_true: float
    f0_true: float
    detected: bool
    v_est: float = np.nan
    r_est: float = np.nan
    t_est: float = np.nan
    v_sig: float = np.nan
    r_sig: float = np.nan
    t_sig: float = np.nan
    peak_snr_db: float = np.nan

    def as_dict(self) -> dict:
        return asdict(self)


def match_events(
    truth: list[EventTruth], events: list[DetectedEvent], t_tol: float = 6.0, tiers: str = "AB"
) -> tuple[list[Match], int]:
    """Greedy one-to-one matching on heard-CPA time. Returns (per-truth matches, number of false positives)."""
    events = [e for e in events if e.tier in tiers]
    used: set[int] = set()
    out: list[Match] = []
    for tr in sorted(truth, key=lambda x: x.r_cpa):
        best, best_d = None, np.inf
        for i, e in enumerate(events):
            if i in used:
                continue
            tol = max(t_tol, 3 * e.fit.sigma()["t_cpa_heard"])
            d = abs(e.fit.t_cpa_heard - tr.t_cpa_heard)
            if d <= tol and d < best_d:
                best, best_d = i, d
        base = dict(
            truth_label=tr.label, r_cpa_true=tr.r_cpa, v_true=tr.speed_cpa, t_heard_true=tr.t_cpa_heard, f0_true=tr.f0
        )
        if best is None:
            out.append(Match(**base, detected=False))
            continue
        used.add(best)
        f = events[best].fit
        s = f.sigma()
        out.append(Match(**base, detected=True, v_est=f.v, r_est=f.r_cpa, t_est=f.t_cpa_heard, v_sig=s["v"],
                         r_sig=s["r_cpa"], t_sig=s["t_cpa_heard"]))  # fmt: skip
    return out, len(events) - len(used)


def summarize(matches: list[Match], n_fp: int, hours: float) -> dict:
    det = [m for m in matches if m.detected]
    out: dict = {"n_truth": len(matches), "n_detected": len(det), "recall": len(det) / max(len(matches), 1),
                 "false_pos_per_hour": n_fp / max(hours, 1e-9)}  # fmt: skip
    if det:
        ev = np.array([(m.v_est - m.v_true) / m.v_true for m in det])
        er = np.array([(m.r_est - m.r_cpa_true) / m.r_cpa_true for m in det])
        et = np.array([m.t_est - m.t_heard_true for m in det])
        zv = np.array([(m.v_est - m.v_true) / m.v_sig for m in det])
        zr = np.array([(m.r_est - m.r_cpa_true) / m.r_sig for m in det])
        zt = np.array([(m.t_est - m.t_heard_true) / m.t_sig for m in det])
        out.update(
            v_rel_err_median=float(np.median(np.abs(ev))),
            r_rel_err_median=float(np.median(np.abs(er))),
            t_err_median_s=float(np.median(np.abs(et))),
            v_rel_err_p90=float(np.percentile(np.abs(ev), 90)),
            r_rel_err_p90=float(np.percentile(np.abs(er), 90)),
            t_err_p90_s=float(np.percentile(np.abs(et), 90)),
            coverage_1sigma={k: float(np.mean(np.abs(z) <= 1)) for k, z in (("v", zv), ("r", zr), ("t", zt))},
            coverage_2sigma={k: float(np.mean(np.abs(z) <= 2)) for k, z in (("v", zv), ("r", zr), ("t", zt))},
        )
    return out
