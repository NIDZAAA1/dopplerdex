"""Lossy codec round-trips through ffmpeg (phone calls, social-media uploads).

Profiles approximate real channels:
  - phone_nb : 8 kHz narrowband, Opus VOIP 12 kbit/s (AMR-NB stand-in; VOIP mode applies
               an adaptive 60-100 Hz high-pass like real telephony stacks)
  - phone_wb : 16 kHz wideband, Opus VOIP 24 kbit/s (AMR-WB / VoLTE stand-in)
  - mp3_64k, aac_96k, opus_music_32k : typical upload/streaming encodes
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

PROFILES: dict[str, dict] = {
    "phone_nb": {"ar": 8000, "args": ["-c:a", "libopus", "-b:a", "12k", "-application", "voip"], "ext": "ogg"},
    "phone_wb": {"ar": 16000, "args": ["-c:a", "libopus", "-b:a", "24k", "-application", "voip"], "ext": "ogg"},
    "mp3_64k": {"ar": None, "args": ["-c:a", "libmp3lame", "-b:a", "64k"], "ext": "mp3"},
    "aac_96k": {"ar": None, "args": ["-c:a", "aac", "-b:a", "96k"], "ext": "m4a"},
    "opus_music_32k": {"ar": None, "args": ["-c:a", "libopus", "-b:a", "32k", "-application", "audio"], "ext": "ogg"},
}


def codec_roundtrip(x: F64, fs: int, profile: str) -> F64:
    """Encode + decode `x` (float, [-1, 1]) and return it at the original rate and length."""
    p = PROFILES[profile]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.f32"
        enc = Path(d) / f"enc.{p['ext']}"
        src.write_bytes(np.asarray(x, dtype=np.float32).tobytes())
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(fs), "-ac", "1", "-i", str(src)]
        if p["ar"]:
            cmd += ["-ar", str(p["ar"])]
        subprocess.run([*cmd, *p["args"], str(enc)], check=True)
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(enc), "-f", "f32le", "-ar", str(fs), "-ac", "1", "-"],
            check=True,
            capture_output=True,
        ).stdout
    y = np.frombuffer(out, dtype=np.float32).astype(np.float64)
    # Codecs add priming delay; align by cross-correlation on a coarse envelope, then trim/pad.
    lag = _estimate_lag(x, y, fs)
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), y])
    return np.pad(y, (0, max(0, x.size - y.size)))[: x.size]


def _estimate_lag(x: F64, y: F64, fs: int, max_lag_s: float = 0.2) -> int:
    n = min(x.size, y.size, int(20 * fs))
    a, b = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    m = int(max_lag_s * fs)
    spec = np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n))
    cc = np.fft.irfft(spec)
    cc = np.concatenate([cc[-m:], cc[: m + 1]])
    return int(np.argmax(cc)) - m
