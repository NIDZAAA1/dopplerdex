# Benchmarks

All benchmark audio is downloaded locally from its publisher; nothing here redistributes recordings.
Ground truth comes from the publisher (recordist map pins, project-published coordinates). Truth is
used only to score results, never as an input.

## Protocol

Same for every system compared: search a 15 km radius around a centre placed 5 km from the truth (bearing
from a hash of the case id, as in hermes' evaluation harness); the recording start time is known to a
stated tolerance (+-30 s by default, wider where clocks are known to be unreliable).

## Real recordings

### Eclipse Soundscapes (`bench/eclipse/`)

AudioMoth recorders deployed for the 2024 total solar eclipse, coordinates and audio published by the
project under CC BY 4.0. Site-hours were chosen by ADS-B traffic before any audio was examined. The
recorders' filename clocks were found to be 5-45 s off, so the time tolerance is +-90 s.

| clip | dopplerdex | hermes (its own harness, published flags) |
|---|---|---|
| es755_20 | **0.03 km** | 0.32 km |
| es736_19 | **0.11 km** | 3.71 km |
| es681_19 | **0.82 km** | 2.87 km |
| es700_19 | 7.25 km | **0.38 km** |
| es783_19 | 5.95 km (truth ranked #2, 0.41 km) | **3.89 km** |
| es783_21 | 5.77 km | no events |

Reproduce: `bench/eclipse/README.md`.

### Public field recordings (`bench/compare/`, `test_data/external/manifest.json`)

The field-recording set from hermes' published evaluation (Freesound CC0, radio aporee public domain).
Thirteen of the 19 recordings contain no aircraft close enough for any system to detect.

| recording | dopplerdex | hermes |
|---|---|---|
| man_hg0903 | 0.44 km | **0.25 km** |
| man_hg0909 | 0.54 km | **0.26 km** |
| muc_ap82011 | **4.79 km** | 13.03 km |
| fai_ap84541 | **5.58 km** | 11.89 km |
| ram_fs807178 | 15.7 km | 19.5 km |
| reference challenge (colsto) | **~0.15 km** | 17.8 km |

The two Manchester answers from dopplerdex agree with each other to ~100 m but sit ~450 m from the
recordist's pin (stated precision about 300 m); the cause is not yet known.

The colsto challenge answer is not published in this repository. To include it locally, create
`test_data/colsto_challenge/truth.local.json` (gitignored):
`{"id": "colsto_challenge", "file": "<audio file name>", "lat": ..., "lon": ..., "start_utc": "...", "loc_precision_m": 100}`.

## Other projects' own benchmarks (`bench/external/`)

- DOPPLERGEIST synthetic discovery controls (14 clips, their generator and evaluator): 8/8 flyovers,
  0 false alarms (DOPPLERGEIST: 8/8, 2 false alarms).
- DOPPLERGEIST four-flyover fixture with decoys: 37 m, correct flights.
- Livermore smartphone recordings (40, detection only): aircraft found in 31/40 with 6 unmatched events
  (DOPPLERGEIST: 35/40 with 859 unmatched events).

See `bench/external/RESULTS.md` for details and caveats.

## Synthetic

`bench/detection_bench.py` and `bench/locate_bench.py` render scenes with the physically based simulator
(`src/dopplerdex/sim`) from real ADS-B tracks, with realistic noise and codecs.
