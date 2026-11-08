"""Broadband 'loud pass' events: loudness-envelope peaks (typically jets) usable as timing events.

A jet passing within a few km produces a broadband loudness peak lasting tens of seconds. Its time is a
heard-CPA estimate (jets are loudest slightly after CPA because of aft-radiated jet noise), with no
usable speed or range information. Paired with one tonal Doppler event it still gives the coarse
search its crucial second timing constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import find_peaks

from dopplerdex.search.loudness import observed_level


@dataclass(frozen=True)
class LoudEvent:
    t_peak: float  # recording time of the loudness maximum (s)
    prominence_db: float
    width_s: float

    @property
    def evidence(self) -> float:
        return float(self.prominence_db * np.sqrt(self.width_s))


def detect_loud_events(x: np.ndarray, fs: float, min_prominence_db: float = 6.0, min_width_s: float = 12.0,
                       max_width_s: float = 180.0, smooth_s: float = 5.0) -> list[LoudEvent]:  # fmt: skip
    """Peaks of the 300-3000 Hz band level above a running low-percentile baseline."""
    t, lv = observed_level(x, fs, 300.0, min(3000.0, 0.45 * fs), frame_s=1.0)
    sm = uniform_filter1d(lv, int(smooth_s))
    base = percentile_filter(sm, 10, size=601, mode="nearest")
    ex = sm - base
    pk, props = find_peaks(ex, prominence=min_prominence_db, width=(min_width_s, max_width_s), distance=30)
    out = [
        LoudEvent(float(t[i]), float(p), float(w))
        for i, p, w in zip(pk, props["prominences"], props["widths"], strict=True)
    ]
    return sorted(out, key=lambda e: -e.evidence)
