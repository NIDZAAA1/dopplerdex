# Data

Everything is cached under `data/` (gitignored). Nothing you record is uploaded anywhere.

## Flight data (adsb.lol)

adsb.lol publishes one archive per day (`globe_history`, GitHub releases, ~3-4 GB, ODbL). dopplerdex
downloads a day once, keeps only the region you need and stores it as hourly Parquet files:

```
data/raw/adsblol/<date>/*.tar.a?                         downloaded archive parts
data/processed/adsb/region=<store>/date=<date>/hour=HH/   prepared track points
```

Regions (`src/dopplerdex/adsb/ingest.py`): `north_america`, `europe`, `world`. Prepare a day with the
web app's *Data* tab or `scripts/ingest_adsb.py`. Some days are missing from the archive; the web app
says so before downloading.

Coverage depends on volunteer receivers: aircraft at low altitude far from a receiver, and aircraft
without ADS-B, are absent. Positions are reported every ~5-7 s (median), with GNSS altitude for >90 %.

## Environment

| what | where cached | notes |
|---|---|---|
| Terrain, Copernicus DEM GLO-90 | `data/cache/dem/` | coarse mosaics for the search, 90 m tiles for refinement |
| Geoid, EGM96 | `data/cache/geoid/` | ellipsoid vs. sea-level heights |
| Weather, Open-Meteo | `data/cache/weather/` | temperature and wind profiles for refinement |
| Aircraft type table | `data/cache/aircraft_types2.json` | ICAO type designator -> propulsion class |

## Your recordings

Uploads in the web app are stored under `data/uploads/` on your machine. Delete that folder to remove
them. Keep private ground truth (where you recorded) in gitignored `*.local.json` or `TRUTH*` files.
