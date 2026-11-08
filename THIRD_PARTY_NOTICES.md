# Third-party notices

dopplerdex itself is licensed under GPL-3.0-or-later (see `LICENSE`). It includes or uses the
following third-party components and data. Python dependencies are installed from PyPI under their
own licences (see `pyproject.toml` / `uv.lock`) and are not redistributed here.

## Bundled code

| component | where | licence |
|---|---|---|
| Leaflet 1.9.4 | `src/dopplerdex/web/static/vendor/leaflet.js`, `leaflet.css` | BSD 2-Clause, see `src/dopplerdex/web/static/vendor/LEAFLET_LICENSE` |

## Data downloaded at run time (not redistributed)

| data | source | licence / terms | attribution |
|---|---|---|---|
| Historical ADS-B tracks | adsb.lol `globe_history` archives (GitHub releases) | Open Database License (ODbL) 1.0 | "Contains data from adsb.lol, available under the ODbL" |
| Aircraft type designators (ICAO Doc 8643 codes) | tar1090-db (`db/icao_aircraft_types2.js`) | per that project | tar1090-db by wiedehopf |
| Terrain | Copernicus DEM GLO-90 (AWS open data) | Copernicus DEM licence | "Copernicus DEM GLO-90, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA" |
| Geoid | EGM96, NGA, via the PROJ data CDN | public domain | NGA EGM96 |
| Weather profiles | Open-Meteo historical API | CC BY 4.0 | "Weather data by Open-Meteo.com" |
| Place-name lookup (optional, web app) | OpenStreetMap Nominatim | ODbL; Nominatim usage policy | "© OpenStreetMap contributors" |
| Map tiles (web app) | Esri basemaps | Esri terms of use | "Tiles © Esri" |

## Benchmark recordings (referenced, never redistributed)

Benchmarks download audio locally from the original publishers. Manifests in `test_data/` and
`bench/` list each recording's source page, recordist and licence:

- Freesound recordings, CC0 1.0 (see `test_data/external/ATTRIBUTION.md`)
- radio aporee ::: maps recordings, public domain
- Eclipse Soundscapes project recordings (Zenodo community `eclipsesoundscapes`), CC BY 4.0
- Livermore aircraft recordings (Zenodo record 23424278), CC BY 4.0

## Other open-source projects

The comparison scripts in `bench/compare` and `bench/external` *invoke* hermes-aircraft-doppler-
geolocation (MIT) and DOPPLERGEIST from a separate local checkout. No code from either project is
included in this repository.
