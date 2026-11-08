"""Static figures (matplotlib) for spectrograms, geometry and Doppler curves."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dopplerdex.audio.spectrogram import Spectrogram  # noqa: E402

PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffb703", "#9d4edd", "#fb8500", "#06d6a0", "#ef476f"]


def plot_spectrogram(
    ax,
    spec: Spectrogram,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
    log_f: bool = False,
):
    db = spec.db()
    if vmin is None:
        vmin = float(np.percentile(db, 30))
    if vmax is None:
        vmax = float(np.percentile(db, 99.7))
    m = ax.pcolormesh(spec.t, spec.f, db, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto", rasterized=True)
    if log_f:
        ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    return m


def overlay_curves(
    ax,
    t,
    curves: Sequence[np.ndarray],
    color: str,
    label: str | None = None,
    lw: float = 1.0,
    ls: str = "--",
    mask=None,
    alpha: float = 0.9,
):
    for k, f in enumerate(curves):
        y = np.where(mask, f, np.nan) if mask is not None else f
        ax.plot(t, y, color=color, lw=lw, ls=ls, alpha=alpha, label=label if k == 0 else None)


def save(fig, path: str | Path, dpi: int = 110):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def dark_fig(nrows=1, ncols=1, figsize=(14, 5), **kw):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor("#0d1117")
    for a in np.atleast_1d(ax).ravel():
        a.set_facecolor("#0d1117")
    return fig, ax
