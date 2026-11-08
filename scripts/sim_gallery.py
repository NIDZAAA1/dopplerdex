"""Render a gallery of synthetic flyover scenarios into a self-contained HTML page.

Usage:  uv run python scripts/sim_gallery.py [--out data/gallery] [--fast]
Each scenario: whitened spectrogram (+ ground-truth curve overlay), geometry, and audio players
(as recorded, and aircraft-only) so you can both *see* and *hear* the physics.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import subprocess
import time
from pathlib import Path

import numpy as np

from dopplerdex.audio.spectrogram import spectrogram, whiten
from dopplerdex.sim.catalog import build_cases
from dopplerdex.sim.codec import codec_roundtrip
from dopplerdex.sim.scenario import SimResult, simulate
from dopplerdex.viz.plots import PALETTE, dark_fig, overlay_curves, plot_spectrogram

FS = 16_000


# ---- figures -------------------------------------------------------------------------


def fig_png(fig, save_to: Path | None = None) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="jpg", dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 82}
    )
    if save_to is not None:
        save_to.write_bytes(buf.getvalue())
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def spectro_figure(audio, res: SimResult, f_max: float, save_to: Path | None = None) -> str:
    sp = whiten(spectrogram(audio, FS, nperseg=8192, hop=1024)).band(25, f_max)
    fig, ax = dark_fig(2, 1, figsize=(15, 8.5), sharex=True)
    for a in ax:
        plot_spectrogram(a, sp, vmin=0, vmax=22)
    ax[0].set_xlabel("")
    ax[0].set_title("whitened spectrogram (what the detector sees)", loc="left", fontsize=10)
    ax[1].set_title("same, with ground-truth tone curves from the physics model", loc="left", fontsize=10)
    for k, tr in enumerate(res.truth):
        n_show = int(np.sum(tr.tone_freqs[:, tr.active].max(axis=1, initial=0) < f_max)) if tr.active.any() else 0
        order = np.argsort(-tr.tone_levels_db)[: min(6, max(n_show, 1))]
        curves = [tr.tone_freqs[i] for i in order if np.nanmax(np.where(tr.active, tr.tone_freqs[i], np.nan)) < f_max]
        overlay_curves(ax[1], tr.control_t, curves, PALETTE[k % len(PALETTE)], tr.label, lw=0.9, mask=tr.active)
        ax[1].axvline(tr.t_cpa_heard, color=PALETTE[k % len(PALETTE)], lw=0.8, ls=":")
    ax[1].legend(loc="upper right", fontsize=9)
    return fig_png(fig, save_to)


def geometry_figure(res: SimResult, save_to: Path | None = None) -> str:
    fig, ax = dark_fig(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for k, (rs, tr) in enumerate(zip(res.rendered, res.truth, strict=True)):
        col = PALETTE[k % len(PALETTE)]
        d = rs.direct
        m = rs.active
        ax[0].scatter(d.src_pos[m, 0] / 1000, d.src_pos[m, 1] / 1000, c=rs.control_t[m], cmap="viridis", s=1.5)
        ax[0].plot(d.src_pos[m, 0] / 1000, d.src_pos[m, 1] / 1000, color=col, lw=0.5, alpha=0.6, label=tr.label)
        i = int(np.argmin(np.where(m, d.rng, np.inf)))
        ax[0].plot([0, d.src_pos[i, 0] / 1000], [0, d.src_pos[i, 1] / 1000], color=col, ls="--", lw=0.8)
        ax[1].plot(rs.control_t[m], d.rng[m] / 1000, color=col, label=f"{tr.label} slant range (km)")
        elev = np.degrees(np.arcsin(np.clip(d.src_pos[:, 2] / d.rng, -1, 1)))
        ax2 = ax[1].twinx() if k == 0 else ax2  # noqa: F821
        ax2.plot(rs.control_t[m], elev[m], color=col, ls=":", lw=1)
        ax2.set_ylabel("elevation angle (deg, dotted)")
    ax[0].plot(0, 0, marker="*", color="white", ms=14)
    ax[0].annotate("mic", (0, 0), (6, 6), textcoords="offset points")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("east (km)")
    ax[0].set_ylabel("north (km)")
    ax[0].set_title("top-down track (colour = heard time), dashed = CPA", loc="left", fontsize=10)
    ax[1].set_xlabel("heard time (s)")
    ax[1].set_ylabel("slant range (km)")
    ax[1].legend(loc="upper left", fontsize=9)
    return fig_png(fig, save_to)


def mp3_b64(x: np.ndarray, gain_to_peak: float | None = None) -> str:
    y = x / (np.max(np.abs(x)) + 1e-12) * gain_to_peak if gain_to_peak else x
    raw = np.clip(y, -1, 1).astype(np.float32).tobytes()
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(FS), "-ac", "1", "-i", "-", "-c:a", "libmp3lame",
         "-b:a", "48k", "-f", "mp3", "-"],
        input=raw, check=True, capture_output=True,
    ).stdout  # fmt: skip
    return base64.b64encode(out).decode()


def truth_table(res: SimResult) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(t.label)}</td><td>{t.aircraft_class}</td><td>{t.f0:.1f}</td>"
        f"<td>{t.speed_cpa:.0f}</td><td>{t.alt_cpa:.0f}</td><td>{t.r_cpa:.0f}</td><td>{t.t_cpa_heard:.1f}</td></tr>"
        for t in res.truth
    )
    return (
        "<table><tr><th>aircraft</th><th>class</th><th>f0 (Hz)</th><th>speed (m/s)</th><th>alt (m)</th>"
        f"<th>CPA slant (m)</th><th>CPA heard (s)</th></tr>{rows}</table>"
    )


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--acc:#4cc9f0;--line:#30363d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 6px}
.lead{color:var(--muted);max-width:820px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 28px}
nav a{color:var(--acc);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:3px 12px;font-size:13px}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 18px 10px;margin:0 0 26px}
h2{font-size:19px;margin:0 0 6px}p{margin:6px 0 12px;color:#c9d1d9}img{width:100%;border-radius:8px;display:block;margin:8px 0}
.audio{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:8px 0 12px}.audio label{font-size:13px;color:var(--muted);display:block}
audio{height:34px}table{border-collapse:collapse;font-size:13px;margin:6px 0 12px}td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}.tag{font-size:12px;color:var(--muted)}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gallery")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections, nav = [], []
    for case in build_cases(args.fast):
        t0 = time.time()
        res = simulate(case.scenario)
        audio = res.audio
        if case.codec:
            audio = codec_roundtrip(audio, FS, case.codec)
        spec_png = spectro_figure(audio, res, case.f_max, out / f"{case.key}_spectrogram.jpg")
        geo_png = geometry_figure(res, out / f"{case.key}_geometry.jpg")
        rec = mp3_b64(audio, gain_to_peak=0.9)
        plane = mp3_b64(res.components["aircraft"], gain_to_peak=0.9)
        dt = time.time() - t0
        print(f"{case.key:24s} {case.scenario.duration:6.0f} s audio  rendered+plotted in {dt:5.1f} s")
        nav.append(f'<a href="#{case.key}">{html.escape(case.title.split(" —")[0])}</a>')
        sections.append(
            f'<section id="{case.key}"><h2>{html.escape(case.title)}</h2><p>{html.escape(case.blurb)}</p>'
            f'<div class="audio"><div><label>as recorded (normalised)</label><audio controls preload="none" '
            f'src="data:audio/mpeg;base64,{rec}"></audio></div><div><label>aircraft only (normalised)</label>'
            f'<audio controls preload="none" src="data:audio/mpeg;base64,{plane}"></audio></div>'
            f'<span class="tag">{case.scenario.duration:.0f} s · seed {case.scenario.seed}'
            f"{' · codec ' + case.codec if case.codec else ''}</span></div>"
            f"{truth_table(res)}"
            f'<img alt="spectrogram" src="data:image/jpeg;base64,{spec_png}">'
            f'<img alt="geometry" src="data:image/jpeg;base64,{geo_png}"></section>'
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Flyover Simulation Gallery</title><style>{CSS}</style></head><body><main>"
        "<h1>Flyover simulation gallery</h1><p class='lead'>Synthetic recordings rendered by the dopplerdex "
        "physics engine: retarded-time Doppler, spherical spreading, ISO 9613-1 absorption, ground reflection, "
        "turbulence, ambient noise and device/codec chains. Dashed curves are ground truth from the same physics "
        "the localiser will use; dotted lines mark when each aircraft's closest approach is <em>heard</em>.</p>"
        f"<nav>{''.join(nav)}</nav>{''.join(sections)}</main></body></html>"
    )
    path = out / "index.html"
    path.write_text(page)
    print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
