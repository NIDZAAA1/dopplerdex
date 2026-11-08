"""Detect aircraft Doppler events in any recording and write a visual report.

uv run python scripts/analyze_recording.py <audio-or-video> [--out data/analysis/<name>] [--temp-c 20]
Outputs: events.json, overview.jpg (whole recording, detected events marked), event_XX.jpg zooms
with the fitted Doppler model overlaid on every harmonic ridge.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dopplerdex.audio.events import DetectConfig, detect_events
from dopplerdex.audio.spectrogram import load_audio, spectrogram, whiten
from dopplerdex.physics.atmosphere import speed_of_sound
from dopplerdex.viz.plots import PALETTE, dark_fig, plot_spectrogram, save

FS = 8000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--temp-c", type=float, default=20.0, help="air temperature for c (until weather is wired in)")
    ap.add_argument("--fmax", type=float, default=800.0)
    args = ap.parse_args()
    src = Path(args.path)
    out = Path(args.out or f"data/analysis/{src.stem[:40]}")
    out.mkdir(parents=True, exist_ok=True)
    x, fs = load_audio(src, FS)
    c = float(speed_of_sound(args.temp_c, 60.0))
    t0 = time.time()
    events, _ = detect_events(x, fs, DetectConfig(c=c))
    print(f"{len(x) / fs / 60:.1f} min of audio, {len(events)} events, detection {time.time() - t0:.1f} s (c={c:.1f})")
    rows = []
    for i, e in enumerate(events):
        f, s = e.fit, e.fit.sigma()
        rows.append({"id": i, "t_start": e.t_start, "t_end": e.t_end, "t_cpa_heard": f.t_cpa_heard,
                     "t_cpa_heard_sigma": s["t_cpa_heard"], "v": f.v, "v_sigma": s["v"], "r_cpa": f.r_cpa,
                     "r_cpa_sigma": s["r_cpa"], "f0": f.f0, "rms_cents": f.rms_cents, "n_ridges": 1 + len(e.companions),
                     "evidence": e.evidence, "lowest_ridge": bool(e.harmonic_hint), "tier": e.tier})  # fmt: skip
        print(f"  #{i:02d} [{e.tier}] heard CPA {f.t_cpa_heard:7.1f}±{s['t_cpa_heard']:.1f}s  v={f.v:5.1f}±{s['v']:4.1f} m/s  "
              f"r={f.r_cpa:6.0f}±{s['r_cpa']:5.0f} m  f0={f.f0:6.1f} Hz  ridges={1 + len(e.companions):2d}  "
              f"evidence={e.evidence:7.0f}")  # fmt: skip
    (out / "events.json").write_text(json.dumps({"source": str(src), "c": c, "events": rows}, indent=1))
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            d = b["t_cpa_heard"] - a["t_cpa_heard"]
            if 880 < d < 960:
                print(
                    f"  pair #{a['id']:02d} -> #{b['id']:02d}: offset {d:.1f} s  (the video's two events were 920 s apart)"
                )

    # ---- overview -------------------------------------------------------------------
    sp = whiten(spectrogram(x, fs, nperseg=4096, hop=2048)).band(35, args.fmax)
    n_rows = int(np.ceil(sp.t[-1] / 900))
    fig, ax = dark_fig(n_rows, 1, figsize=(20, 3.6 * n_rows))
    ax = np.atleast_1d(ax)
    for k in range(n_rows):
        seg = sp.time_slice(k * 900, (k + 1) * 900)
        plot_spectrogram(ax[k], seg, vmin=0, vmax=14)
        for i, e in enumerate(events):
            if k * 900 <= e.fit.t_cpa_heard < (k + 1) * 900:
                col = PALETTE[i % len(PALETTE)]
                ax[k].axvline(e.fit.t_cpa_heard, color=col, lw=1.0, ls=":")
                ax[k].text(e.fit.t_cpa_heard + 3, args.fmax * 0.93, f"#{i:02d}", color=col, fontsize=9)
                ax[k].plot(e.ridge.t, e.ridge.f, color=col, lw=0.8, alpha=0.8)
    save(fig, out / "overview.jpg", dpi=70)

    # ---- per-event zooms --------------------------------------------------------------
    full = whiten(spectrogram(x, fs, nperseg=4096, hop=512))
    for i, e in enumerate(events):
        lo, hi = e.t_start - 20, e.t_end + 20
        fr = np.concatenate([e.ridge.f, *[cp.f for cp in e.companions]])
        fmax = min(float(np.percentile(fr, 99)) * 1.25, fs / 2 * 0.95)
        seg = full.time_slice(lo, hi).band(30, fmax)
        fig, axz = dark_fig(1, 2, figsize=(18, 5.2))
        for a in axz:
            plot_spectrogram(a, seg, vmin=0, vmax=16)
        tt = np.linspace(lo, hi, 600)
        for r in [e.ridge, *e.companions]:
            ratio = np.median(r.f / e.fit.curve(r.t))
            axz[1].plot(tt, ratio * e.fit.curve(tt), color="#4cc9f0", lw=0.9, ls="--")
        f, s = e.fit, e.fit.sigma()
        axz[1].axvline(f.t_cpa_heard, color="#ffb703", lw=0.8, ls=":")
        axz[0].set_title(f"event #{i:02d}", loc="left", fontsize=10)
        axz[1].set_title(f"fit: v={f.v:.1f}±{s['v']:.1f} m/s  r_cpa={f.r_cpa:.0f}±{s['r_cpa']:.0f} m  heard CPA "
                         f"{f.t_cpa_heard:.1f}±{s['t_cpa_heard']:.1f} s", loc="left", fontsize=10)  # fmt: skip
        save(fig, out / f"event_{i:02d}.jpg", dpi=75)
    print("report in", out)


if __name__ == "__main__":
    main()
