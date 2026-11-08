# Tests on the other projects' own benchmarks (2026-11-08)

## 1. DOPPLERGEIST synthetic discovery controls (14 clips, their generator + their evaluator)
Noise, stationary tone, non-aircraft sweep, moving harmonic source, overlapping pair, truncated 8 s clip, MP3;
development + held-out seeds. Scored by `dopplergeist.evaluation` (temporal IoU >= 0.3).

| system | flyovers found | misses | false alarms | time / clip |
|---|---|---|---|---|
| dopplerdex | 8 / 8 | 0 | **0** | 0.7 s |
| DOPPLERGEIST (afdb2bd) | 8 / 8 | 0 | 2 (both truncated 8 s clips) | 4.5 s |

## 2. hermes at its published settings (its own harness `scripts/eval_batch.py`, `--det broadband=true --set temper_extra=4`)
Commit ff25280 is not in the public history; current 0.3.1 with the published flags reproduces the published
Manchester numbers (0.25 / 0.26 km vs published 0.19 / 0.33 km). Our earlier harness called hermes' CLI and
got 2.73 / 1.76 km on the same clips, i.e. it under-represented hermes; all hermes numbers below come from
its own harness, and our search centre now uses hermes' exact formula (md5 bearing), so both systems search
the identical 15 km area with the same +-30 s (+-30 min for approximate recordist times) window.

| recording | dopplerdex | hermes (own harness) |
|---|---|---|
| colsto_challenge | **0.15 km** | 17.76 km |
| man_hg0903 | 0.51 km | **0.25 km** |
| man_hg0909 | 0.94 km | **0.26 km** |
| muc_ap82011 | **5.08 km** | 13.03 km |
| fai_ap84541 | **5.56 km** | 11.89 km |
| ram_fs807178 | 15.75 km | 19.51 km |
| other 13 | no answer (no aircraft curves) | no answer (no events) |

Answered 6 / 6; < 1 km: 3 / 2; median of answered 3.0 km / 12.5 km.

## 3. DOPPLERGEIST four-flyover fixture (12 toy flights in three regions, decoys mimic events 1-2)
Request coverage = CONUS, known clock, flat 122 m ground, zero geoid (fixture's operator terrain).

| dopplerdex version | error | truth rank | matched flights |
|---|---|---|---|
| before explained-events ranking | 884 km (decoy) | 3 | decoy pair |
| after | **37 m** | 1 | 2 of the 4 true flights (pair), all 4 events explained |

## 4. Livermore: 40 real smartphone recordings (Zenodo 23424278, CC BY 4.0), detection only
No sample-zero clock or recorder coordinates are published, so this measures detection only. Event matches the
annotated aircraft if its ridge overlaps the annotated frequency band (+-3 %, harmonic ratios 1/3..3) for half
the band. Same rule for both systems.

| system | clips with the aircraft found | events proposed | unmatched events | time / clip |
|---|---|---|---|---|
| dopplerdex | 31 / 40 | 37 | 6 (16 %) | 1.1 s |
| DOPPLERGEIST (afdb2bd) | 35 / 40 | 1,102 | 859 (78 %) | 4.6 s |

DOPPLERGEIST proposes ~28 events per 1-minute clip, so the loose band rule credits some hits by chance; ours is
far more selective (higher precision, lower recall). DOPPLERGEIST's own doc (older e8cbb08) reports 136 groups
in 36 clips.
