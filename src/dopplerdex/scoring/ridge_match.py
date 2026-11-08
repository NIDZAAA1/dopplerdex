"""Score (place, T0, flight A, flight B) hypotheses by how well real flight geometry reproduces the
detected Doppler ridges.

For a hypothesis the microphone is at P (ground elevation + mic height). For each detected ridge
point (recording time t, frequency f), the reception time is T0 + t; the emission time t_e is solved
by Newton iteration on the flight's (linearly interpolated) track, and the predicted log-frequency is
log f0 + log D, D = 1 / (1 - n.v / c). The unknown f0 of each ridge (harmonic number x source
frequency) is a constant in log-f, removed by a weighted mean. The residual RMS (cents) measures the
curve-shape agreement. Event A sweeps a small clock offset dt; event B is locked to the same dt, so its
timing relative to A is purely geometric (cf. the reference video's "combined locked scoring").
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from numpy.typing import NDArray

from dopplerdex.audio.events import DetectedEvent

F64 = NDArray[np.float64]
CENTS = 1200.0 / np.log(2.0)


@dataclass
class RidgeSet:
    """All ridges of one detected event, flattened: times, log-frequency, weights, segment ids."""

    @staticmethod
    def empty() -> RidgeSet:
        z = np.zeros(0)
        return RidgeSet(z, z, z, np.zeros(0, np.int64), 0)

    t: F64
    lf: F64
    w: F64
    seg: NDArray[np.int64]
    n_seg: int

    @staticmethod
    def from_event(ev: DetectedEvent, max_points_per_ridge: int = 80, min_score: float = 0.5) -> RidgeSet:
        ts, lfs, ws, segs = [], [], [], []
        for k, r in enumerate([ev.ridge, *ev.companions]):
            step = max(1, r.t.size // max_points_per_ridge)
            sel = slice(0, None, step)
            ts.append(r.t[sel])
            lfs.append(np.log(r.f[sel]))
            ws.append(np.clip(r.score[sel], min_score, 20.0))
            segs.append(np.full(ts[-1].size, k, np.int64))
        return RidgeSet(np.concatenate(ts), np.concatenate(lfs), np.concatenate(ws), np.concatenate(segs), len(ts))


@nb.njit(cache=True, fastmath=True)
def _predict_logd(tr_t, tr_e, tr_n, tr_u, tr_ve, tr_vn, tr_vu, i0, i1, t_rec, c, out_logd, out_ok):
    """Retarded-time Doppler for a microphone at the ENU origin along one track [i0, i1)."""
    j = i0
    for k in range(t_rec.size):
        tr = t_rec[k]
        te = tr - 5.0
        ok = False
        ld = 0.0
        for _ in range(8):
            # locate te
            if te < tr_t[i0] or te > tr_t[i1 - 1]:
                ok = False
                break
            while j < i1 - 2 and tr_t[j + 1] < te:
                j += 1
            while j > i0 and tr_t[j] > te:
                j -= 1
            a = (te - tr_t[j]) / (tr_t[j + 1] - tr_t[j])
            se = tr_e[j] + a * (tr_e[j + 1] - tr_e[j])
            sn = tr_n[j] + a * (tr_n[j + 1] - tr_n[j])
            su = tr_u[j] + a * (tr_u[j + 1] - tr_u[j])
            ve = tr_ve[j] + a * (tr_ve[j + 1] - tr_ve[j])
            vn = tr_vn[j] + a * (tr_vn[j + 1] - tr_vn[j])
            vu = tr_vu[j] + a * (tr_vu[j + 1] - tr_vu[j])
            dist = np.sqrt(se * se + sn * sn + su * su)
            nv = -(se * ve + sn * vn + su * vu) / dist  # n = (O - S)/|O - S|, O = 0
            g = te + dist / c - tr
            gp = 1.0 - nv / c
            te -= g / gp
            ld = -np.log(1.0 - nv / c)
            ok = True
            if abs(g) < 1e-4:
                break
        out_logd[k] = ld
        out_ok[k] = ok


@nb.njit(cache=True, fastmath=True)
def _segment_rms(lf, logd, ok, w, seg, n_seg, clip):
    """Weighted RMS (in log units) after removing a per-segment constant; residuals clipped at +-clip."""
    tot_w = 0.0
    tot = 0.0
    n_ok = 0
    for s in range(n_seg):
        sw = 0.0
        sm = 0.0
        for k in range(lf.size):
            if seg[k] == s and ok[k]:
                sw += w[k]
                sm += w[k] * (lf[k] - logd[k])
        if sw <= 0:
            continue
        mu = sm / sw
        # one robust re-centring pass
        sw2 = 0.0
        sm2 = 0.0
        for k in range(lf.size):
            if seg[k] == s and ok[k]:
                d = lf[k] - logd[k] - mu
                if abs(d) < 3 * clip:
                    sw2 += w[k]
                    sm2 += w[k] * d
        if sw2 > 0:
            mu += sm2 / sw2
        for k in range(lf.size):
            if seg[k] == s and ok[k]:
                d = lf[k] - logd[k] - mu
                if d > clip:
                    d = clip
                elif d < -clip:
                    d = -clip
                tot += w[k] * d * d
                tot_w += w[k]
                n_ok += 1
    if tot_w <= 0:
        return 1e9, 0
    return np.sqrt(tot / tot_w), n_ok


@nb.njit(parallel=True, cache=True)
def score_hypotheses(
    lat_p, lon_p, z_mic, t0, fid_a, fid_b,  # per hypothesis
    tr_t, tr_lat, tr_lon, tr_alt, tr_gs, tr_trk, tr_vu, fid_start, fid_end,  # tracks
    a_t, a_lf, a_w, a_seg, a_nseg, b_t, b_lf, b_w, b_seg, b_nseg,  # ridges
    c, dts, clip, min_valid,
):  # fmt: skip
    n = lat_p.size
    rms_a = np.full(n, 1e9)
    rms_b = np.full(n, 1e9)
    best_dt = np.zeros(n)
    for h in nb.prange(n):
        coslat = np.cos(np.radians(lat_p[h]))
        best = (1e18, 1e9, 1e9, 0.0)
        # local ENU tracks of both flights (copied slices)
        ia0, ia1 = fid_start[fid_a[h]], fid_end[fid_a[h]]
        ib0, ib1 = fid_start[fid_b[h]], fid_end[fid_b[h]]
        na, nb_ = ia1 - ia0, ib1 - ib0
        if na < 3 or nb_ < 3:
            continue
        ae = np.empty(na)
        an = np.empty(na)
        au = np.empty(na)
        ave = np.empty(na)
        avn = np.empty(na)
        avu = np.empty(na)  # noqa: E702
        for i in range(na):
            q = ia0 + i
            ae[i] = (tr_lon[q] - lon_p[h]) * 111320.0 * coslat
            an[i] = (tr_lat[q] - lat_p[h]) * 111320.0
            au[i] = tr_alt[q] - z_mic[h]
            tr_r = np.radians(tr_trk[q])
            ave[i] = tr_gs[q] * np.sin(tr_r)
            avn[i] = tr_gs[q] * np.cos(tr_r)
            avu[i] = tr_vu[q]
        be = np.empty(nb_)
        bn = np.empty(nb_)
        bu = np.empty(nb_)
        bve = np.empty(nb_)
        bvn = np.empty(nb_)
        bvu = np.empty(nb_)  # noqa: E702
        for i in range(nb_):
            q = ib0 + i
            be[i] = (tr_lon[q] - lon_p[h]) * 111320.0 * coslat
            bn[i] = (tr_lat[q] - lat_p[h]) * 111320.0
            bu[i] = tr_alt[q] - z_mic[h]
            tr_r = np.radians(tr_trk[q])
            bve[i] = tr_gs[q] * np.sin(tr_r)
            bvn[i] = tr_gs[q] * np.cos(tr_r)
            bvu[i] = tr_vu[q]
        ta = tr_t[ia0:ia1]
        tb = tr_t[ib0:ib1]
        logd_a = np.empty(a_t.size)
        ok_a = np.zeros(a_t.size, np.bool_)
        logd_b = np.empty(b_t.size)
        ok_b = np.zeros(b_t.size, np.bool_)
        for dt in dts:
            _predict_logd(ta, ae, an, au, ave, avn, avu, 0, na, t0[h] + dt + a_t, c, logd_a, ok_a)
            ra, nva = _segment_rms(a_lf, logd_a, ok_a, a_w, a_seg, a_nseg, clip)
            if nva < min_valid * a_t.size:
                continue
            if b_t.size > 0:  # timing-only partners (loud passes) have no ridges to compare
                _predict_logd(tb, be, bn, bu, bve, bvn, bvu, 0, nb_, t0[h] + dt + b_t, c, logd_b, ok_b)
                rb, nvb = _segment_rms(b_lf, logd_b, ok_b, b_w, b_seg, b_nseg, clip)
                if nvb < min_valid * b_t.size:
                    continue
            else:
                rb = 0.0
            cost = ra * ra * a_t.size + rb * rb * b_t.size
            if cost < best[0]:
                best = (cost, ra, rb, dt)
        rms_a[h] = best[1]
        rms_b[h] = best[2]
        best_dt[h] = best[3]
    return rms_a * CENTS, rms_b * CENTS, best_dt
