"""Refuse to commit recordings, private ground truth, secrets or large files.

Checks every file git would commit (tracked + staged), by CONTENT as well as by name, so a recording
renamed to .jpeg or .txt is still caught. Used by the pre-commit hook and by CI:

    python scripts/check_repo_hygiene.py            # all tracked files
    python scripts/check_repo_hygiene.py --staged   # only what is staged (pre-commit)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000  # larger files need an explicit allowance below
ALLOW_LARGE = {"uv.lock", "LICENSE"}
MEDIA_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff", ".caf", ".amr",
             ".3gp", ".mp4", ".mov", ".mkv", ".webm", ".avi"}  # fmt: skip
# magic numbers of audio/video containers
MAGIC = [
    (0, b"RIFF", "RIFF (WAV/AVI)"),
    (0, b"ID3", "MP3 (ID3)"),
    (0, b"\xff\xfb", "MP3 frame"),
    (0, b"\xff\xf3", "MP3 frame"),
    (0, b"\xff\xf1", "AAC (ADTS)"),
    (0, b"\xff\xf9", "AAC (ADTS)"),
    (0, b"fLaC", "FLAC"),
    (0, b"OggS", "Ogg/Opus"),
    (0, b"FORM", "AIFF"),
    (0, b"caff", "CAF"),
    (0, b"#!AMR", "AMR"),
    (4, b"ftyp", "MP4/M4A/MOV"),
    (0, b"\x1a\x45\xdf\xa3", "Matroska/WebM"),
]
PRIVATE_NAME = re.compile(r"(^|/)(TRUTH[^/]*|[^/]*\.local\.json|[^/]*\.private\.[^/]*|\.env(\.[^/]*)?)$")
ABS_USER_PATH = re.compile(rb"/(Users|home)/[A-Za-z0-9._-]+/")


def files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and Path(f).is_file()]


def main() -> int:
    problems = []
    for f in files("--staged" in sys.argv):
        p = Path(f)
        if p.name == ".env.example":
            continue
        if p.suffix.lower() in MEDIA_EXT:
            problems.append(f"{f}: audio/video file extension")
        head = p.read_bytes()[:16]
        for off, sig, what in MAGIC:
            if head[off : off + len(sig)] == sig:
                problems.append(f"{f}: content looks like {what}")
                break
        if PRIVATE_NAME.search(f):
            problems.append(f"{f}: private ground truth / secrets file")
        size = p.stat().st_size
        if size > MAX_BYTES and p.name not in ALLOW_LARGE:
            problems.append(f"{f}: {size / 1e6:.1f} MB (limit {MAX_BYTES / 1e6:.0f} MB)")
        if f != "scripts/check_repo_hygiene.py" and size < 5_000_000 and ABS_USER_PATH.search(p.read_bytes()):
            problems.append(f"{f}: contains an absolute personal path (/Users/... or /home/...)")
    if problems:
        print("Repository hygiene check FAILED:\n  " + "\n  ".join(problems))
        print("Recordings and private truth must never be committed (see CONTRIBUTING.md).")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
