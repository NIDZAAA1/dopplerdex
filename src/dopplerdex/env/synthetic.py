"""Synthetic-world switch for fixtures that define their own terrain and datum.

Some test fixtures (e.g. DOPPLERGEIST's four-flyover case) render on constant ground with a zero geoid.
use_flat_world() makes every terrain/geoid lookup return those constants; it is never enabled by the
normal pipeline, the web app or the comparison on real recordings.
"""

from __future__ import annotations

FLAT: dict[str, float] | None = None


def use_flat_world(ground_m: float, geoid_m: float = 0.0) -> None:
    global FLAT
    FLAT = {"ground": float(ground_m), "geoid": float(geoid_m)}


def clear() -> None:
    global FLAT
    FLAT = None
