# Eclipse Soundscapes test set (real recordings, published coordinates, 2024-04-08)

Source: Eclipse Soundscapes project, AudioMoth recorders deployed by volunteer data collectors during the
2024 total solar eclipse; site coordinates and audio published by the project on Zenodo under
**CC BY 4.0** (community `eclipsesoundscapes`; per-site records listed in the generated `data/raw/eclipse/manifest.json`). Audio is not
redistributed here: `fetch_clip.py` extracts one UTC hour per site straight from the Zenodo zips.

Selection (before any audio was examined): `rank_sites.py` ranks site-hours by close ADS-B passes
(< 3 km, < 1,500 m above the site) on 2024-04-08, the only day of the deployment that adsb.lol archived.
Nine site-hours were chosen; three had no audio in that hour (recorder stopped) and were dropped.

Clock: the filenames claim time-chime-set UTC, but at the true sites the best-fitting clock offset was
-5 ... -45 s (Carbondale sites about -40 s), outside a +-30 s window. Both systems were therefore given
+-90 s. Location precision of the published coordinates: taken as 50 m.

## Results (+-90 s, 15 km search radius around a centre 5 km off the truth, identical for both)

| clip | site | dopplerdex | hermes (own harness, published flags) |
|---|---|---|---|
| es755_20 | San Antonio area | **0.03 km** | 0.32 km |
| es736_19 | Carbondale, IL | **0.11 km** | 3.71 km |
| es681_19 | Carbondale, IL | **0.82 km** | 2.87 km |
| es700_19 | Carbondale, IL | 7.25 km | **0.38 km** |
| es783_19 | San Antonio area | 5.95 km (truth ranked #2, 0.41 km) | **3.89 km** |
| es783_21 | San Antonio area | 5.77 km | no events |

< 1 km: dopplerdex 3 / 6, hermes 2 / 6.

Lessons fed back into the system (applied to every recording):
- pairs are spread over several anchor events (one non-ADS-B anchor no longer sinks the search) and
  busy recordings get 5 pairs: es681 no answer -> 0.82 km.
- es783_19: two places along one straight approach corridor explain 4-5 passes each; the truth is the
  runner-up. Not tuned further (would be fitting this clip).
