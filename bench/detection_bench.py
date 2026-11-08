"""Monte-Carlo benchmark of event detection + Doppler parameter estimation.

uv run python bench/detection_bench.py --n 64 [--workers 8] [--tag note]
Writes bench/results/detection_<stamp>.json, a figure, and appends a row to bench/history.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dopplerdex.audio.events import DetectConfig, detect_events
from dopplerdex.audio.spectrogram import resample
from dopplerdex.eval.detection import match_events, summarize
from dopplerdex.eval.random_scenarios import random_scenario
from dopplerdex.sim.codec import codec_roundtrip
from dopplerdex.sim.scenario import simulate

ROOT = Path(__file__).resolve().parent


def run_one(seed: int) -> dict:
    scn, codec = random_scenario(seed)
    res = simulate(scn)
    a = res.audio
    if codec:
        a = codec_roundtrip(a, int(scn.fs), codec)
    x = resample(a, int(scn.fs), 8000)
    t0 = time.time()
    events, _ = detect_events(x, 8000, DetectConfig(c=scn.atmosphere.c))  # c known from weather data
    dt = time.time() - t0
    matches, n_fp = match_events(res.truth, events, tiers="A")
    matches_ab, n_fp_ab = match_events(res.truth, events, tiers="AB")
    return {
        "recall_ab": float(np.mean([m.detected for m in matches_ab])) if matches_ab else np.nan,
        "n_fp_ab": n_fp_ab,
        "seed": seed,
        "codec": codec,
        "texture": scn.texture is not None,
        "detect_s": dt,
        "duration": scn.duration,
        "n_fp": n_fp,
        "matches": [m.as_dict() for m in matches],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(run_one, range(args.seed0, args.seed0 + args.n)))
    from dopplerdex.eval.detection import Match

    matches = [Match(**m) for r in runs for m in r["matches"]]
    hours = sum(r["duration"] for r in runs) / 3600
    summ = summarize(matches, sum(r["n_fp"] for r in runs), hours)
    # recall by distance band
    bands = [(0, 1000), (1000, 2000), (2000, 3500), (3500, 7000)]
    summ["recall_by_r_cpa"] = {
        f"{a}-{b}": float(
            np.mean([m.detected for m in matches if a <= m.r_cpa_true < b])
            if any(a <= m.r_cpa_true < b for m in matches)
            else np.nan
        )
        for a, b in bands
    }
    summ["by_codec"] = {}
    for cd in sorted({str(r["codec"]) for r in runs}):
        ms = [Match(**m) for r in runs if str(r["codec"]) == cd for m in r["matches"]]
        summ["by_codec"][cd] = float(np.mean([m.detected for m in ms]))
    summ["tierAB_recall"] = float(np.nanmean([r["recall_ab"] for r in runs]))
    summ["tierAB_false_pos_per_hour"] = sum(r["n_fp_ab"] for r in runs) / hours
    summ["detect_seconds_per_audio_hour"] = sum(r["detect_s"] for r in runs) / hours
    summ["wall_s"] = time.time() - t0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"detection_{stamp}.json"
    out.write_text(json.dumps({"summary": summ, "runs": runs, "args": vars(args)}, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    hist = ROOT / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "stamp",
                    "bench",
                    "tag",
                    "n",
                    "recall",
                    "fp_per_hour",
                    "v_rel_med",
                    "r_rel_med",
                    "t_med_s",
                    "cov2_v",
                    "cov2_r",
                    "cov2_t",
                ]
            )
        w.writerow(
            [
                stamp,
                "detection",
                args.tag,
                args.n,
                f"{summ['recall']:.3f}",
                f"{summ['false_pos_per_hour']:.2f}",
                f"{summ.get('v_rel_err_median', np.nan):.3f}",
                f"{summ.get('r_rel_err_median', np.nan):.3f}",
                f"{summ.get('t_err_median_s', np.nan):.2f}",
                *(f"{summ['coverage_2sigma'][k]:.2f}" for k in "vrt"),
            ]
        )
    _plot(matches, out.with_suffix(".png"))
    print("wrote", out)


def _plot(matches, path: Path) -> None:
    from dopplerdex.viz.plots import dark_fig, save

    fig, ax = dark_fig(1, 3, figsize=(16, 4.5))
    r = np.array([m.r_cpa_true for m in matches])
    d = np.array([m.detected for m in matches])
    ax[0].scatter(
        r, d + np.random.default_rng(0).uniform(-0.05, 0.05, d.size), s=10, c=np.where(d, "#06d6a0", "#ef476f")
    )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("true CPA slant range (m)")
    ax[0].set_yticks([0, 1], ["missed", "detected"])
    det = [m for m in matches if m.detected]
    if det:
        ax[1].errorbar(
            [m.r_cpa_true for m in det],
            [m.r_est for m in det],
            yerr=[m.r_sig for m in det],
            fmt="o",
            ms=3,
            color="#4cc9f0",
            alpha=0.7,
        )
        lim = [200, 8000]
        ax[1].plot(lim, lim, color="white", lw=0.7)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("true CPA range (m)")
        ax[1].set_ylabel("estimated (±1σ)")
        z = np.concatenate(
            [
                [(m.v_est - m.v_true) / m.v_sig for m in det],
                [(m.r_est - m.r_cpa_true) / m.r_sig for m in det],
                [(m.t_est - m.t_heard_true) / m.t_sig for m in det],
            ]
        )
        ax[2].hist(np.clip(z, -6, 6), bins=36, color="#ffb703", density=True)
        xx = np.linspace(-6, 6, 200)
        ax[2].plot(xx, np.exp(-(xx**2) / 2) / np.sqrt(2 * np.pi), color="white", lw=1)
        ax[2].set_xlabel("normalised error (est-true)/σ  — should match N(0,1)")
    save(fig, path)


if __name__ == "__main__":
    main()
