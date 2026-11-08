"""Catalogue of named synthetic scenarios (used by the gallery, tests and benchmarks)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from dopplerdex.audio.spectrogram import load_audio
from dopplerdex.sim.noise import AmbientConfig, NoiseProfile
from dopplerdex.sim.render import Atmosphere, Device, Ground
from dopplerdex.sim.scenario import Scenario, manoeuvring_pass, straight_pass
from dopplerdex.sim.sources import db_to_pa

FS = 16_000
CHALLENGE = Path(__file__).resolve().parents[3] / Path(
    "test_data/colsto_challenge/unknown_location_raw_audio_4CxVxuMVkMc_f140.m4a"
)


@dataclass
class Case:
    key: str
    title: str
    blurb: str
    scenario: Scenario
    f_max: float = 1200.0
    codec: str | None = None


@lru_cache(maxsize=1)
def _challenge_profile() -> NoiseProfile | None:
    if not CHALLENGE.exists():
        return None
    x, fs = load_audio(CHALLENGE, FS)
    x = x[int(2400 * fs) : int(3000 * fs)]  # a 10-minute stretch of the real recording
    return NoiseProfile.fit_band_levels(x, fs)


def challenge_texture(dev: Device) -> NoiseProfile | None:
    """Band-level statistics of the real challenge recording, scaled to pascals for this device."""
    prof = _challenge_profile()
    if prof is None:
        return None
    return prof.scaled(20 * np.log10(db_to_pa(dev.full_scale_db) / np.sqrt(2)))


def build_cases(fast: bool) -> list[Case]:
    dur = 0.5 if fast else 1.0
    dev = Device()
    cases = [
        Case(
            "c172_close",
            "Cessna 172 — close pass, quiet suburb",
            "Piston single, 2-blade prop at ~2450 RPM → blade-passing ≈ 82 Hz plus harmonics. 1.2 km slant "
            "range at CPA, 600 m up, 55 m/s. Note the comb of Doppler curves and the U-shaped interference "
            "fringes in the broadband noise (ground reflection / Lloyd's mirror).",
            Scenario([straight_pass("C172", 90 * dur, 1200, 600, 55, 70)], duration=200 * dur, fs=FS, seed=1),
        ),
        Case(
            "kingair_far_realnoise",
            "King Air, 3.2 km away, in the *real* challenge background",
            "Turboprop at 100 m/s, CPA slant 3.2 km — like the video's event 1 (f0 ≈ 113 Hz). The background is "
            "synthesised from the band-level statistics of the real 1-hour challenge recording (texture "
            "resampling), so it has the same noise floor, slope and non-stationarity.",
            Scenario(
                [straight_pass("BE20", 110 * dur, 3200, 1500, 100, 250, side=-1)],
                duration=240 * dur,
                fs=FS,
                seed=2,
                texture=challenge_texture(dev),
                ambient=AmbientConfig(background_db=0, wind_db=0, birds_per_min=1, steady_tones=()),
            ),
        ),
        Case(
            "r44_helicopter",
            "Robinson R44 helicopter",
            "Main rotor 2 blades × 400 RPM → 13 Hz (sub-audible) with many harmonics (blade slap); tail rotor ≈ 81 Hz "
            "fundamental. Helicopters give dense harmonic ladders.",
            Scenario([straight_pass("R44", 80 * dur, 900, 300, 45, 300)], duration=170 * dur, fs=FS, seed=3),
        ),
        Case(
            "a320_approach",
            "Airbus A320 on approach (jet)",
            "Mostly broadband roar, plus a fan blade-passing tone near 2 kHz. Jets are hard to use from tones at "
            "long range (≈5–10 dB/km absorption at 1–2 kHz), but the ground-reflection fringes sweep with "
            "geometry — the basis of the broadband method.",
            Scenario(
                [manoeuvring_pass("A320", 0, (-9000, 400, 1100), 260 * dur, 72.0, 90.0, -3.8)],
                duration=220 * dur,
                fs=FS,
                seed=4,
            ),
            f_max=2600,
        ),
        Case(
            "c172_turning",
            "Cessna 172 orbiting near the mic (non-ideal curve)",
            "A standard-rate 360° turn. The curve no longer looks like a textbook Doppler 'S' — exactly the "
            "information the video's final scoring exploits: real flight geometry leaves a fingerprint.",
            Scenario(
                [
                    manoeuvring_pass(
                        "C172",
                        0,
                        (-3500, -800, 700),
                        200 * dur,
                        55.0,
                        np.where(
                            np.arange(0, 200 * dur + 1) < 60 * dur,
                            90.0,
                            90.0 + 3.0 * (np.arange(0, 200 * dur + 1) - 60 * dur),
                        ),
                        0.0,
                    )
                ],
                duration=200 * dur,
                fs=FS,
                seed=5,
            ),
        ),
        Case(
            "two_events",
            "Two flyovers, one recording",
            "An SR22 (3-blade, 6-cyl) then a Caravan turboprop. The *time offset* between the two heard CPAs is "
            "the key constraint of the search (920 s in the video; shorter here).",
            Scenario(
                [
                    straight_pass("SR22", 60 * dur, 1800, 900, 75, 20),
                    straight_pass("C208", 300 * dur, 2500, 1200, 85, 200, side=-1),
                ],
                duration=380 * dur,
                fs=FS,
                seed=6,
            ),
        ),
        Case(
            "strong_wind",
            "Same pass, 15 m/s crosswind",
            "Wind changes the effective sound speed along each ray, skewing the curve slightly and shifting the "
            "heard CPA time — why the pipeline uses real weather (HRRR / ERA5).",
            Scenario(
                [straight_pass("BE20", 90 * dur, 1500, 800, 90, 0)],
                duration=180 * dur,
                fs=FS,
                seed=7,
                atmosphere=Atmosphere(wind=(15.0, 0.0, 0.0), turbulence_db=3.0),
                ambient=AmbientConfig(wind_db=58, wind_gustiness=8),
            ),
        ),
        Case(
            "phone_call",
            "Through a phone call (narrowband VOIP codec)",
            "The two-event-style King Air pass encoded at 8 kHz / 12 kbit/s Opus VOIP (an AMR-NB stand-in, with "
            "its adaptive ~60–100 Hz high-pass). The fundamental is attenuated but higher harmonics survive.",
            Scenario([straight_pass("BE20", 90 * dur, 1800, 900, 95, 300)], duration=180 * dur, fs=FS, seed=8),
            codec="phone_nb",
        ),
        Case(
            "indoor_mic",
            "Indoors-ish: hard ground, low mic, loud room tones",
            "Mic 0.5 m above a rigid floor (reflection fringes move to higher frequency), louder machinery lines "
            "and mains hum — the curve is still there under the clutter.",
            Scenario(
                [straight_pass("PA28", 80 * dur, 1000, 500, 52, 140)],
                duration=170 * dur,
                fs=FS,
                seed=9,
                ground=Ground(mic_height=0.5, flow_resistivity=None),
                ambient=AmbientConfig(
                    background_db=45, mains_db=48, steady_tones=((120.0, 46.0), (180.0, 40.0), (305.0, 38.0))
                ),
            ),
        ),
    ]
    return cases
