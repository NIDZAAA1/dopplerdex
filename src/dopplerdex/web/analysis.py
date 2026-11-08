"""Step 1 of the web flow: analyse a recording and say whether (and how fast) it can be located."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from scipy.signal import welch

from dopplerdex.audio.events import DetectConfig, DetectedEvent, detect_events
from dopplerdex.audio.loud_events import detect_loud_events
from dopplerdex.audio.spectrogram import load_audio, spectrogram, whiten
from dopplerdex.physics.atmosphere import speed_of_sound
from dopplerdex.search.loudness import observed_level

CONUS_AREA_KM2 = 8.08e6
SEC_PER_WINDOW_CONUS = 150.0  # measured: one 2 h start-time window, whole CONUS, busy daytime traffic


def _jpg(fig) -> str:
    import matplotlib.pyplot as plt

    b = io.BytesIO()
    fig.savefig(b, format="jpg", dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 82})
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def mains_hum(x: np.ndarray, fs: float) -> dict:
    """Which electricity grid frequency (50 or 60 Hz) is audible, if any (a coarse regional clue)."""
    f, p = welch(x, fs, nperseg=int(2 ** np.ceil(np.log2(8 * fs))))

    def snr(f0: float) -> float:
        tot = 0.0
        for k in (1, 2, 3):
            band = (f > k * f0 - 0.8) & (f < k * f0 + 0.8)
            ref = (f > k * f0 - 8) & (f < k * f0 + 8) & ~band
            if band.any() and ref.any():
                tot = max(tot, 10 * np.log10(p[band].max() / np.median(p[ref])))
        return float(tot)

    s50, s60 = snr(50.0), snr(60.0)
    grid = None
    if max(s50, s60) > 8 and abs(s50 - s60) > 4:
        grid = 50 if s50 > s60 else 60
    hint = {50: "Europe, Africa, most of Asia, Australia, parts of South America",
            60: "North America, parts of South America, Taiwan, South Korea, western Japan, Saudi Arabia"}  # fmt: skip
    return {"grid_hz": grid, "snr50_db": round(s50, 1), "snr60_db": round(s60, 1), "hint": hint.get(grid)}


def noise_profile(x: np.ndarray, fs: float) -> dict:
    t, lv = observed_level(x, fs, 300.0, min(3000.0, 0.45 * fs))
    med = np.median(lv)
    transient = float(np.mean(lv > med + 10))
    return {"transient_fraction": round(transient, 3), "clipping_fraction": round(float(np.mean(np.abs(x) > 0.98)), 5),
            "dynamic_range_db": round(float(np.percentile(lv, 95) - np.percentile(lv, 5)), 1)}  # fmt: skip


def _event_dict(e: DetectedEvent) -> dict:
    s = e.fit.sigma()
    return {"t": round(e.fit.t_cpa_heard, 1), "tier": e.tier, "v": round(e.fit.v, 1), "v_sigma": round(s["v"], 1),
            "r": round(e.fit.r_cpa), "r_sigma": round(s["r_cpa"]), "f0": round(e.fit.f0, 1), "ridges": 1 + len(e.companions),
            "t_start": round(e.t_start, 1), "t_end": round(e.t_end, 1)}  # fmt: skip


def verdict(events: list[DetectedEvent], loud: list, duration: float, noise: dict) -> dict:
    air = [e for e in events if e.tier in ("A", "B")]
    tier_a = [e for e in air if e.tier == "A"]
    ground = [e for e in events if e.tier == "G"]
    anchor_t = max(air, key=lambda e: (e.tier == "A", e.evidence)).fit.t_cpa_heard if air else None
    partners_loud = [le for le in loud if anchor_t is None or abs(le.t_peak - anchor_t) > 60]
    reasons, tips = [], []
    if len(tier_a) >= 2 or (len(tier_a) >= 1 and len(air) + len(partners_loud) >= 3):
        grade, searchable = "good", True
        reasons.append(
            f"{len(air)} aircraft Doppler curve(s) ({len(tier_a)} well constrained) - enough for a pair search."
        )
    elif len(air) >= 2 or (len(air) == 1 and partners_loud):
        grade, searchable = "fair", True
        reasons.append("Two timing events are available (an aircraft curve plus another curve or a loud pass), but "
                       "speed/distance estimates are weak, so the result may be ambiguous.")  # fmt: skip
    elif len(air) == 1:
        grade, searchable = "poor", True
        reasons.append("Only one aircraft curve: a single-aircraft search can run, but the answer may slide along "
                       "that flight's path or mirror to its other side unless the start time is known closely.")  # fmt: skip
        tips.append(
            "Give the start time as precisely as you can (a few minutes or better); record longer to catch more passes."
        )
    else:
        grade, searchable = "unusable", False
        reasons.append("No aircraft Doppler curves were found.")
    if loud:
        reasons.append(f"{len(loud)} loud broadband pass(es) (usually jets) detected.")
    if ground:
        reasons.append(f"{len(ground)} road/rail vehicle pass(es) recognised and ignored.")
    if duration < 600:
        tips.append(f"The recording is {duration / 60:.1f} min long; 30-60 min gives many more aircraft passes.")
    if noise["transient_fraction"] > 0.15:
        tips.append("Lots of short loud bursts (handling, wind, nearby activity): keep the phone still and sheltered.")
    if noise["clipping_fraction"] > 1e-3:
        tips.append("The recording clips (too loud); move the phone away from loud sources.")
    return {"grade": grade, "searchable": searchable, "reasons": reasons, "tips": tips,
            "n_air": len(air), "n_tier_a": len(tier_a), "n_ground": len(ground), "n_loud": len(loud)}  # fmt: skip


def estimate_seconds(area_km2: float, window_hours: float, n_pairs: int, n_loud_pairs: int, data_state: str,
                     n_single: int = 0) -> dict:  # fmt: skip
    windows = max(1, math.ceil(window_hours / 2.0))
    per_window = SEC_PER_WINDOW_CONUS * max(area_km2 / CONUS_AREA_KM2, 0.02)
    search = (n_pairs + n_loud_pairs + n_single) * windows * per_window
    consistency = 0.6 * (60 * n_pairs + 300 * n_loud_pairs + 350 * n_single)
    refine = 45.0
    data = {"ready": 0.0, "raw": 120.0, "remote": 420.0}.get(data_state, 0.0)
    mid = search + consistency + refine + data
    return {"low": round(0.5 * mid), "high": round(2.0 * mid), "mid": round(mid),
            "breakdown": {"data": round(data), "search": round(search), "consistency": round(consistency), "refine": refine}}  # fmt: skip


def analyse(path: str, temp_c: float = 15.0, progress=None) -> dict:
    def rep(f: float, m: str) -> None:
        if progress:
            progress(f, m)

    rep(0.05, "decoding audio")
    x, fs = load_audio(path, 8000)
    duration = len(x) / fs
    c = float(speed_of_sound(temp_c, 70.0))
    rep(0.2, "detecting aircraft Doppler curves")
    events, _ = detect_events(x, fs, DetectConfig(c=c))
    rep(0.6, "detecting loud passes, mains hum and noise")
    loud = detect_loud_events(x, fs)
    hum = mains_hum(x, fs)
    noise = noise_profile(x, fs)
    v = verdict(events, loud, duration, noise)
    rep(0.8, "drawing the spectrogram")
    from dopplerdex.viz.plots import PALETTE, dark_fig, plot_spectrogram

    sp = whiten(spectrogram(x, fs, nperseg=8192, hop=1024 if duration < 1800 else 2048)).band(35, 1200)
    fig, ax = dark_fig(1, 1, figsize=(16, 5.2))
    plot_spectrogram(ax, sp, vmin=0, vmax=14)
    for k, e in enumerate(events):
        col = "#9aa7b5" if e.tier == "G" else PALETTE[k % len(PALETTE)]
        tt = np.linspace(e.t_start, e.t_end, 300)
        for r in [e.ridge, *e.companions]:
            ratio = np.median(r.f / e.fit.curve(r.t))
            ax.plot(tt, ratio * e.fit.curve(tt), color=col, lw=1.0, ls="--")
        ax.text(e.t_start, 1150, f"{'road' if e.tier == 'G' else 'aircraft'} {k + 1}", color=col, fontsize=9)
    for le in loud:
        ax.axvline(le.t_peak, color="#ffb547", lw=0.8, ls=":")
    ax.set_title("detected aircraft curves (dashed), loud passes (dotted)", loc="left", fontsize=10)
    img = _jpg(fig)
    rep(1.0, "done")
    return {"duration_s": round(duration, 1), "events": [_event_dict(e) for e in events],
            "loud": [{"t": round(le.t_peak, 1), "prominence_db": round(le.prominence_db, 1), "width_s": round(le.width_s)} for le in loud],
            "hum": hum, "noise": noise, "verdict": v, "spectrogram_jpg": img, "c": round(c, 1)}  # fmt: skip
