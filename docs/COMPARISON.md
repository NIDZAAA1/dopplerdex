# Comparison with other open-source projects

How dopplerdex compares with the two other open-source implementations of acoustic aircraft
geolocation we know of, on real recordings with independently known locations. Raw per-recording
results: [`bench/results/comparison.csv`](../bench/results/comparison.csv).

| project | version tested | how it was run |
|---|---|---|
| **dopplerdex** | this repository (Nov 2026) | `bench/compare/run_compare.py` |
| [hermes-aircraft-doppler-geolocation](https://github.com/mojomast/hermes-aircraft-doppler-geolocation) (MIT) | 0.3.1 (`14b4b0c`) | its **own** evaluation harness `scripts/eval_batch.py` with its published settings (`--det broadband=true --set temper_extra=4`) |
| [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST) | `afdb2bd` | its CLI (`analyze --online`), with a one-line fix for a crash present on every input ([PATCHES.md](../bench/compare/PATCHES.md)) |

## Protocol

Identical for every system, following hermes' published evaluation protocol:

- search a **15 km radius** around a centre placed **5 km from the truth** (bearing derived from the
  recording id, hermes' formula), so no search grid is aligned on the answer;
- the recording start time is given with a tolerance: **+-30 s** (field recordings), **+-30 min** where
  the recordist gave only an approximate time, **+-90 s** for Eclipse Soundscapes (their recorder clocks
  were found to be 5-45 s off);
- the same audio files and the same ADS-B day archives (adsb.lol) for all systems;
- the true location is used only to score the answer.

## Test sets

- **Field recordings:** the 18 Freesound (CC0) and radio aporee (public domain) recordings from hermes'
  published evaluation, with the recordists' map pins as truth (stated precision 50-1,500 m).
- **Reference challenge:** the recording from colsto's video challenge (answer kept private, see
  [BENCHMARKS.md](BENCHMARKS.md)).
- **Eclipse Soundscapes:** 6 one-hour AudioMoth recordings from 2024-04-08 with project-published
  coordinates (CC BY 4.0), chosen by ADS-B traffic near the site before any audio was examined.

## Results

### Summary (25 recordings)

| | dopplerdex | hermes | DOPPLERGEIST |
|---|---|---|---|
| recordings answered | **12** | 11 | 0 of 19 (not run on Eclipse) |
| answers within 1 km | **6** | 4 | 0 |
| answers within 3 km | **6** | 5 | 0 |
| median error of answers | **2.8 km** | 3.7 km | - |
| closer to the truth, head to head (11 recordings both answered) | **7** | 4 | - |

### Per recording (error in km; "-" = no answer)

| set | recording | dopplerdex | hermes | DOPPLERGEIST |
|---|---|---|---|---|
| challenge | colsto challenge | **0.15** | 17.76 | - |
| field | man_hg0903 | 0.44 | **0.25** | - |
| field | man_hg0909 | 0.54 | **0.26** | - |
| field | muc_ap82011 | **4.79** | 13.03 | - |
| field | fai_ap84541 | **5.58** | 11.89 | - |
| field | ram_fs807178 | **15.74** | 19.51 | - |
| eclipse | es755_20 | **0.03** | 0.32 | not run |
| eclipse | es736_19 | **0.11** | 3.71 | not run |
| eclipse | es681_19 | **0.82** | 2.87 | not run |
| eclipse | es700_19 | 7.25 | **0.38** | not run |
| eclipse | es783_19 | 5.95 | **3.89** | not run |
| eclipse | es783_21 | **5.77** | - | not run |

The remaining 13 field recordings got no answer from any system: none detected a usable aircraft event
in them (one has no ADS-B archive for its day).

### Other projects' own benchmarks

| benchmark | dopplerdex | other project |
|---|---|---|
| DOPPLERGEIST synthetic discovery controls (14 clips; their generator and scorer) | 8/8 flyovers, **0** false alarms | DOPPLERGEIST: 8/8, 2 false alarms |
| DOPPLERGEIST four-flyover fixture with decoy flights | **37 m**, correct flights | - |
| Livermore smartphone recordings (40, detection only) | aircraft found in 31/40, 6 unmatched events | DOPPLERGEIST: 35/40, 859 unmatched events |

## How to read this

- **Small sample.** Only 11-12 recordings produce an answer at all; differences of one or two
  recordings change the summary. Treat this as indicative, not definitive.
- **Not a held-out test for dopplerdex.** Several general improvements to dopplerdex (multipath events,
  explained-event ranking, pair selection) were developed while looking at these recordings. Each was
  checked not to make other recordings worse, but these are development sets. hermes likewise tuned its
  settings on some of them (the Manchester recordings are in its tuning split).
- **Truth precision.** Recordist pins can be off by hundreds of metres (Manchester: about 300 m), so
  sub-kilometre differences there are not meaningful.
- **DOPPLERGEIST abstains by design.** It returns "inconclusive" unless its own consistency checks pass;
  its documentation states that real-world localisation accuracy has not been established. Its
  detector was compared separately on its own benchmarks above.
- **Written by an interested party.** The harness and this page were written by the dopplerdex authors.
  The other systems were run through their own published entry points and settings, and the full
  per-recording results are published so anyone can check or rerun them. Corrections are welcome.

## Reproduce

```bash
# field recordings: audio listed with sources in test_data/external/manifest.json
uv run python bench/compare/run_compare.py --systems ours
# Eclipse Soundscapes: see bench/eclipse/README.md
uv run python bench/eclipse/make_manifest.py      # after fetching clips (bench/eclipse/README.md)
uv run python bench/compare/run_compare.py --cases data/raw/eclipse/manifest.json --systems ours
```

hermes and DOPPLERGEIST are run from separate local checkouts in `reference/` (not included);
`bench/compare/run_compare.py` and `bench/eclipse/run_hermes.sh` show the exact commands.
