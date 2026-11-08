"""Ridge tracking on a log-frequency feature map.

Two stages:
  1. Peak linking (McAulay-Quatieri style): per-frame local maxima above a threshold are
     linked into tracks by predicted-frequency continuity, tolerating short gaps.
  2. Viterbi refinement inside a corridor around each linked track: the globally best
     smooth path (quadratic penalty on frequency jumps), with sub-bin parabolic refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dopplerdex.audio.features import LogFreqMap

F64 = NDArray[np.float64]


@dataclass
class Track:
    frames: list[int] = field(default_factory=list)
    bins: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    misses: int = 0

    def predict(self, frame: int) -> float:
        if len(self.bins) < 4:
            return self.bins[-1]
        k = min(len(self.bins), 12)
        fr = np.asarray(self.frames[-k:], dtype=float)
        bn = np.asarray(self.bins[-k:])
        slope = np.polyfit(fr, bn, 1)[0]
        return self.bins[-1] + slope * (frame - self.frames[-1])


@dataclass(frozen=True)
class Ridge:
    t: F64  # times (s)
    bin: F64  # sub-bin positions on the log-f axis
    f: F64  # Hz (harmonic-sum fundamental equivalent)
    score: F64  # feature value along the ridge
    mean_score: float

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])


def _frame_peaks(col: NDArray, thr: float, min_sep: int) -> NDArray[np.int64]:
    c = col
    m = (c[1:-1] > thr) & (c[1:-1] >= c[:-2]) & (c[1:-1] > c[2:])
    idx = np.flatnonzero(m) + 1
    if idx.size <= 1 or min_sep <= 1:
        return idx
    # non-maximum suppression within min_sep bins
    order = idx[np.argsort(-c[idx])]
    keep: list[int] = []
    for i in order:
        if all(abs(i - j) >= min_sep for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), dtype=np.int64)


def link_peaks(
    fm: LogFreqMap,
    thr: float = 3.5,
    tol_bins: float = 3.0,
    max_gap_frames: int = 16,
    min_sep_bins: int = 6,
) -> list[Track]:
    z = fm.z
    active: list[Track] = []
    done: list[Track] = []
    for j in range(z.shape[1]):
        peaks = _frame_peaks(z[:, j], thr, min_sep_bins)
        used = np.zeros(peaks.size, dtype=bool)
        # strongest tracks get first pick
        for tr in sorted(active, key=lambda t: -np.mean(t.scores[-8:])):
            if peaks.size == 0:
                tr.misses += 1
                continue
            pred = tr.predict(j)
            tol = tol_bins * (1.0 + 0.25 * tr.misses)
            d = np.abs(peaks - pred)
            d[used] = np.inf
            k = int(np.argmin(d))
            if d[k] <= tol:
                used[k] = True
                tr.frames.append(j)
                tr.bins.append(float(peaks[k]))
                tr.scores.append(float(z[peaks[k], j]))
                tr.misses = 0
            else:
                tr.misses += 1
        still = []
        for tr in active:
            (done if tr.misses > max_gap_frames else still).append(tr)
        active = still
        for k in np.flatnonzero(~used):
            active.append(Track([j], [float(peaks[k])], [float(z[peaks[k], j])]))
    return done + active


def viterbi_refine(
    fm: LogFreqMap, tr: Track, corridor: int = 8, jump_penalty: float = 0.6, clip: float = 15.0
) -> Ridge:
    """Best smooth path inside +-corridor bins of the (interpolated) linked track."""
    f0, f1 = tr.frames[0], tr.frames[-1]
    frames = np.arange(f0, f1 + 1)
    centre = np.interp(frames, tr.frames, tr.bins)
    nb = 2 * corridor + 1
    rows = np.clip(
        np.round(centre).astype(int)[None, :] + np.arange(-corridor, corridor + 1)[:, None], 0, fm.z.shape[0] - 1
    )
    obs = np.minimum(fm.z[rows, frames[None, :]], clip).astype(np.float64)  # (nb, T)
    steps = np.arange(-3, 4)
    trans = -jump_penalty * steps.astype(float) ** 2
    T = frames.size
    acc = obs[:, 0].copy()
    back = np.zeros((nb, T), dtype=np.int8)
    shift = np.round(np.diff(centre)).astype(int)  # corridor recentres as it moves
    for t in range(1, T):
        prev = np.full(nb, -np.inf)
        arg = np.zeros(nb, dtype=np.int8)
        for s, pen in zip(steps, trans, strict=True):
            # moving from corridor row r' at t-1 to row r at t: absolute bin change = (r - r') + shift
            src = np.arange(nb) + shift[t - 1] - s
            ok = (src >= 0) & (src < nb)
            cand = np.full(nb, -np.inf)
            cand[ok] = acc[src[ok]] + pen
            better = cand > prev
            prev[better] = cand[better]
            arg[better] = s
        acc = prev + obs[:, t]
        back[:, t] = arg
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(acc))
    for t in range(T - 1, 0, -1):
        path[t - 1] = path[t] + shift[t - 1] - back[path[t], t]
        path[t - 1] = min(max(path[t - 1], 0), nb - 1)
    absbin = rows[path, np.arange(T)].astype(float)
    # sub-bin parabolic refinement on the feature map
    zb = fm.z
    ib = absbin.astype(int)
    lo = np.clip(ib - 1, 0, zb.shape[0] - 1)
    hi = np.clip(ib + 1, 0, zb.shape[0] - 1)
    a, b, c = zb[lo, frames], zb[ib, frames], zb[hi, frames]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0.0)
    absbin = absbin + np.clip(off, -0.5, 0.5)
    score = zb[ib, frames].astype(float)
    return Ridge(fm.t[frames], absbin, fm.freq_of(absbin), score, float(np.mean(score)))


def stitch_ridges(
    ridges: list[Ridge], max_gap_s: float = 10.0, tol_oct: float = 0.03, slope_window_s: float = 4.0
) -> list[Ridge]:
    """Join ridge fragments that continue each other (a line lost for a few seconds and re-acquired).

    B continues A if B starts within max_gap_s after A ends (or overlaps by < 3 s), and A's end,
    linearly extrapolated in log-f over the gap, lands within tol_oct of B's start. Greedy chaining,
    strongest continuation first.
    """
    rs = sorted(ridges, key=lambda r: r.t[0])
    used = [False] * len(rs)
    out: list[Ridge] = []
    for i in range(len(rs)):
        if used[i]:
            continue
        cur = rs[i]
        used[i] = True
        while True:
            best, best_err = None, tol_oct
            m = cur.t >= cur.t[-1] - slope_window_s
            lf = np.log2(cur.f[m])
            slope = np.polyfit(cur.t[m], lf, 1)[0] if m.sum() >= 3 else 0.0
            for j in range(len(rs)):
                if used[j]:
                    continue
                b = rs[j]
                gap = b.t[0] - cur.t[-1]
                if gap > max_gap_s or gap < -3.0:
                    continue
                k = min(b.t.size, max(3, int(slope_window_s / max(np.median(np.diff(b.t)), 1e-3))))
                pred = np.log2(cur.f[-1]) + slope * (b.t[0] - cur.t[-1])
                err = abs(np.median(np.log2(b.f[:k])) - pred)
                if err < best_err:
                    best, best_err = j, err
            if best is None:
                break
            b = rs[best]
            used[best] = True
            keep = b.t > cur.t[-1]
            t = np.concatenate([cur.t, b.t[keep]])
            score = np.concatenate([cur.score, b.score[keep]])
            cur = Ridge(t, np.concatenate([cur.bin, b.bin[keep]]), np.concatenate([cur.f, b.f[keep]]), score,
                        float(np.mean(score)))  # fmt: skip
        out.append(cur)
    return out
