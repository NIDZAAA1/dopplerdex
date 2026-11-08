"""Coarse search: which (place, recording-start) hypotheses explain two detected events?

Geometry: a ground point P hears the closest approach of a (locally straight) flight at the track
sample whose position S_i is such that P lies in the plane through S_i perpendicular to the velocity.
For level-ish flight that plane meets the ground in the line through the sub-aircraft point,
perpendicular to the ground track. All ground points at CPA slant range r from sample i are the two
points on that line at horizontal offset h = sqrt(r^2 - H^2) (H = height above ground). So for an
event with slant-range band [r_lo, r_hi] each speed-compatible sample emits two short line segments
of candidate points, each with an implied recording start T0 = t_i + r / c - t_event.

Two events agree on a hypothesis if they produce candidates in the same place with the same T0.
That is found with a hash join on (2 km cell, T0 bin), probing neighbouring bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from numpy.typing import NDArray

from dopplerdex.search.tracks import TrackSamples

M_PER_DEG = 111_320.0
LAT_REF = 38.0  # fixed reference latitude for the coarse cell grid (consistent across events)
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class EventConstraint:
    """What the search uses from one detected event: a Gaussian over (ground speed v, CPA slant range r,
    heard-CPA recording time t), conditioned per candidate on the aircraft's actual speed and range.

    v and r are typically ~0.99 correlated in a Doppler fit (steepness ~ v^2 / r), so conditioning on
    the ADS-B speed shrinks the range band several-fold. Unknown wind is modelled as extra variance on
    the speed (ADS-B gives ground speed; the acoustic fit sees roughly the air-relative speed).
    """

    mean: tuple[float, float, float]  # (v, r, t_heard)
    cov: tuple[tuple[float, ...], ...]  # 3x3
    k: float = 3.0  # gate width in (conditional) sigmas
    wind_sigma: float = 10.0  # m/s
    r_margin: float = 250.0  # m, model error (altitude datum, wind refraction)
    t_margin: float = 3.0  # s, model error (sound speed / wind on the travel time)
    r_max: float = 12_000.0
    classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    label: str = ""

    @staticmethod
    def from_fit(fit, label: str = "", **kw) -> EventConstraint:
        mu, C = fit.constraint_gaussian()
        return EventConstraint(tuple(map(float, mu)), tuple(map(tuple, C.tolist())), label=label, **kw)

    @staticmethod
    def from_values(v, v_sigma, r, r_sigma, t, t_sigma, rho_vr=0.0, label: str = "", **kw) -> EventConstraint:
        C = np.diag([v_sigma**2, r_sigma**2, t_sigma**2])
        C[0, 1] = C[1, 0] = rho_vr * v_sigma * r_sigma
        return EventConstraint((v, r, t), tuple(map(tuple, C.tolist())), label=label, **kw)

    def coefficients(self) -> dict[str, float]:
        mv, mr, mt = self.mean
        C = np.array(self.cov, dtype=np.float64)
        cvv = C[0, 0] + self.wind_sigma**2
        s_v = np.sqrt(cvv)
        a_r = C[0, 1] / cvv
        s_r = np.sqrt(max(C[1, 1] - C[0, 1] ** 2 / cvv, 1.0))
        S = np.array([[cvv, C[0, 1]], [C[0, 1], C[1, 1]]])
        st = np.array([C[2, 0], C[2, 1]])
        b = np.linalg.solve(S, st)
        s_t = np.sqrt(max(C[2, 2] - b @ st, 0.01))
        return {"mv": mv, "mr": mr, "mt": mt, "s_v": s_v, "a_r": a_r, "s_r": s_r, "b_v": b[0], "b_r": b[1], "s_t": s_t}

    @property
    def t_heard(self) -> float:
        return self.mean[2]

    @property
    def t_sigma_cond(self) -> float:
        return float(np.hypot(self.coefficients()["s_t"], self.t_margin / 3.0))


@dataclass
class Candidates:
    cell: NDArray[np.int64]
    t0: F64  # implied recording start (unix s)
    fid: NDArray[np.int32]
    lat: F64
    lon: F64
    r: F64

    def __len__(self) -> int:
        return self.cell.size


def cell_id(lat: F64, lon: F64, cell_m: float) -> NDArray[np.int64]:
    row = np.floor(lat * M_PER_DEG / cell_m).astype(np.int64)
    col = np.floor(lon * M_PER_DEG * np.cos(np.radians(LAT_REF)) / cell_m).astype(np.int64)
    return row * 1_000_003 + col


@nb.njit(cache=True)
def _count(ok, h_lo, h_hi, step):
    n = 0
    for i in range(ok.size):
        if ok[i]:
            n += 2 * (int((h_hi[i] - h_lo[i]) / step) + 1)
    return n


@nb.njit(cache=True)
def _emit(ok, t, lat, lon, H, trk, gs, fid, h_lo, h_hi, step, c, mv, mr, mt, b_v, b_r,
          out_lat, out_lon, out_t0, out_fid, out_r):  # fmt: skip
    k = 0
    for i in range(ok.size):
        if not ok[i]:
            continue
        tr = np.radians(trk[i])
        pe, pn = np.cos(tr), -np.sin(tr)  # unit vector perpendicular to the ground track (east, north)
        coslat = np.cos(np.radians(lat[i]))
        m = int((h_hi[i] - h_lo[i]) / step) + 1
        for side in (-1.0, 1.0):
            for j in range(m):
                h = h_lo[i] + j * step
                r = np.sqrt(h * h + H[i] * H[i])
                # heard-time expected for this (speed, range) under the event's Gaussian
                t_ev = mt + b_v * (gs[i] - mv) + b_r * (r - mr)
                out_lat[k] = lat[i] + side * h * pn / 111320.0
                out_lon[k] = lon[i] + side * h * pe / (111320.0 * coslat)
                out_t0[k] = t[i] + r / c - t_ev
                out_fid[k] = fid[i]
                out_r[k] = r
                k += 1
    return k


def generate(ts: TrackSamples, elev: F64, ev: EventConstraint, c: float, step: float, cell_m: float,
             t0_range: tuple[float, float], dedupe_bin_s: float | None = None) -> Candidates:  # fmt: skip
    """Candidate points for one event from all compatible track samples."""
    co = ev.coefficients()
    H = np.maximum(ts.alt_msl - elev, 1.0)
    dv = ts.gs - co["mv"]
    ok = (np.abs(dv) <= ev.k * co["s_v"]) & np.isin(ts.cls, np.array(ev.classes, np.int8))
    r_mid = co["mr"] + co["a_r"] * dv
    half = ev.k * co["s_r"] + ev.r_margin
    r_lo = np.clip(r_mid - half, 30.0, ev.r_max)
    r_hi = np.clip(r_mid + half, 30.0, ev.r_max)
    ok &= (r_hi > H) & (r_hi > r_lo)
    r_lo = np.maximum(r_lo, H)
    h_lo = np.sqrt(np.maximum(r_lo**2 - H**2, 0.0))
    h_hi = np.sqrt(np.maximum(r_hi**2 - H**2, 0.0))
    t0_approx = ts.t + r_mid / c - co["mt"]
    ok &= (t0_approx >= t0_range[0] - 120) & (t0_approx <= t0_range[1] + 120)
    n = _count(ok, h_lo, h_hi, step)
    out_lat, out_lon, out_t0, out_r = (np.empty(n) for _ in range(4))
    out_fid = np.empty(n, np.int32)
    k = _emit(ok, ts.t, ts.lat, ts.lon, H, ts.trk, ts.gs, ts.fid, h_lo, h_hi, step, c, co["mv"], co["mr"], co["mt"],
              co["b_v"], co["b_r"], out_lat, out_lon, out_t0, out_fid, out_r)  # fmt: skip
    keep = (out_t0[:k] >= t0_range[0]) & (out_t0[:k] < t0_range[1])
    la, lo = out_lat[:k][keep], out_lon[:k][keep]
    cand = Candidates(cell_id(la, lo, cell_m), out_t0[:k][keep], out_fid[:k][keep], la, lo, out_r[:k][keep])
    if dedupe_bin_s:
        key = np.stack(
            [cand.cell, cand.fid.astype(np.int64), np.floor(cand.t0 / dedupe_bin_s).astype(np.int64)], axis=1
        )
        _, idx = np.unique(key, axis=0, return_index=True)
        cand = Candidates(cand.cell[idx], cand.t0[idx], cand.fid[idx], cand.lat[idx], cand.lon[idx], cand.r[idx])
    return cand


@dataclass
class PairMatches:
    a_idx: NDArray[np.int64]  # indices into candidates A
    b_idx: NDArray[np.int64]
    dt0: F64

    def __len__(self) -> int:
        return self.a_idx.size


def _join_chunk(A: Candidates, ks: NDArray[np.int64], order: NDArray[np.int64], B: Candidates, bsel: NDArray[np.int64],
                tol: float) -> tuple[NDArray[np.int64], NDArray[np.int64], F64]:  # fmt: skip
    bins_b = np.floor(B.t0[bsel] / tol).astype(np.int64)
    ai, bi = [], []
    for d in (-1, 0, 1):
        key_b = B.cell[bsel] * 4_000_000 + ((bins_b + d) % 4_000_000)
        lo = np.searchsorted(ks, key_b, side="left")
        hi = np.searchsorted(ks, key_b, side="right")
        cnt = hi - lo
        has = np.flatnonzero(cnt > 0)
        if has.size == 0:
            continue
        reps = cnt[has]
        starts = np.repeat(lo[has], reps)
        offs = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        ai.append(order[starts + offs])
        bi.append(bsel[np.repeat(has, reps)])
    if not ai:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0)
    a_idx, b_idx = np.concatenate(ai), np.concatenate(bi)
    dt0 = B.t0[b_idx] - A.t0[a_idx]
    keep = (np.abs(dt0) <= tol) & (A.fid[a_idx] != B.fid[b_idx])
    a_idx, b_idx, dt0 = a_idx[keep], b_idx[keep], dt0[keep]
    if a_idx.size == 0:
        return a_idx, b_idx, dt0
    # compact: one match per (cell, flight A, flight B), the one with the best timing agreement
    key = np.stack([A.cell[a_idx], A.fid[a_idx].astype(np.int64), B.fid[b_idx].astype(np.int64)], axis=1)
    o = np.lexsort((np.abs(dt0), key[:, 2], key[:, 1], key[:, 0]))
    key, a_idx, b_idx, dt0 = key[o], a_idx[o], b_idx[o], dt0[o]
    first = np.ones(key.shape[0], dtype=bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    return a_idx[first], b_idx[first], dt0[first]


def join_pairs(A: Candidates, B: Candidates, tol: float, chunk: int = 2_000_000) -> PairMatches:
    """All (cell, flight A, flight B) hypotheses with |T0_a - T0_b| <= tol (different flights).

    B is processed in chunks and every chunk is compacted to unique (cell, fid_a, fid_b) before the
    next, so memory scales with the number of distinct hypotheses, not the raw many-to-many matches.
    """
    bins_a = np.floor(A.t0 / tol).astype(np.int64)
    key_a = A.cell * 4_000_000 + (bins_a % 4_000_000)
    order = np.argsort(key_a, kind="stable")
    ks = key_a[order]
    out_a, out_b, out_d = [], [], []
    for s0 in range(0, len(B), chunk):
        bsel = np.arange(s0, min(s0 + chunk, len(B)))
        a, b, d = _join_chunk(A, ks, order, B, bsel, tol)
        out_a.append(a)
        out_b.append(b)
        out_d.append(d)
    if not out_a:
        z = np.zeros(0, np.int64)
        return PairMatches(z, z, np.zeros(0))
    return PairMatches(np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_d))
