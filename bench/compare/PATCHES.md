# Patches applied to compared systems

Only crash fixes, never changes to localisation logic.

## DOPPLERGEIST (afdb2bd, 2026-11-06)
`src/dopplergeist/jobs.py`, `execute.<locals>.progress`: the `analyze` job crashed on every input at audio intake with
`TypeError: progress() takes 2 positional arguments but 3 were given`, because `automatic.py` passes
`(stage, fraction, message)` positionally (via `_decode_progress`). Patch: accept an optional positional `message`
and forward it as the existing `message=` field. Behaviour is otherwise identical.

## hermes (14b4b0c, 2026-11-03)
No code changes. Its ADS-B crop cache is pre-filled with its own `adsb.fetch_tracks(..., tar_parts=...)` from the same
local adsb.lol prod-0 archives the other systems use (the CLI only streams from GitHub).
