# dopplerdex

**Where was this recorded?** dopplerdex estimates the location of an audio recording from the aircraft
you can (or can barely) hear in it. Every passing aircraft leaves a Doppler curve in the spectrogram,
and its shape depends on exactly where the microphone was relative to the flight path. Historical
ADS-B data says where every aircraft was at every second, so the curves can be matched to flights
and to the one place on the map where they all fit.

> **Use it responsibly.** A precise location can reveal where someone lives. Only analyse recordings
> you made yourself or have explicit permission to analyse. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## How it works

1. **Detect.** Tonal Doppler curves (propellers, rotors, engines) are tracked in a whitened
   log-frequency spectrogram and fitted with the full retarded-time Doppler model: speed, closest
   distance and time of closest approach, with honest uncertainties. Jets without a usable tone are
   detected from the ground-reflection (Lloyd's mirror) delay curve in the cepstrum.
2. **Search.** For every flight in the area and time window, the places that would have heard each
   detected pass form thin strips beside its track. Pairs of events must agree on place *and*
   recording start time; a hash join finds those agreements among millions of possibilities.
3. **Score.** Surviving hypotheses are scored against the real flight paths (turns, climbs, speed
   changes), then against the whole recording: how many detected events they explain, whether
   predicted loud passes are loud and predicted propeller curves present.
4. **Refine.** The winner is optimised continuously with terrain (Copernicus GLO-90), the geoid,
   weather (Open-Meteo wind and temperature) and every explained aircraft at once.

A single aircraft can also be used, with a weaker (strip-shaped) answer, when the start time is known.

## Results

On real recordings with independently known locations (details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| test set | recordings with an answer | answers < 1 km | best |
|---|---|---|---|
| Eclipse Soundscapes (US, 2024-04-08, published recorder coordinates) | 6 / 6 | 3 | 28 m |
| Public field recordings (Freesound / radio aporee) + reference challenge | 6 / 19 | 3 | ~150 m |

Most failures are recordings with no aircraft close enough to hear, or aircraft that do not broadcast
ADS-B. Read the limitations below before trusting a result.

## Quick start

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), ffmpeg.

```bash
git clone https://github.com/NIDZAAA1/dopplerdex && cd dopplerdex
uv sync
uv run pytest                      # test suite (no downloads needed)
uv run dopplerdex-web              # web app at http://127.0.0.1:8765
```

The web app walks through everything: upload a recording, give a rough area and time, check whether
the audio is usable (with a compute-time estimate), download the flight data for that day, search,
and see the result on a map. A **Learn** tab explains the method with an interactive Doppler lab.

Command line:

```bash
# what aircraft are in this recording? (spectrogram report with fitted curves)
uv run python scripts/analyze_recording.py my_recording.m4a --out data/analysis/mine

# prepare one day of flight data for a region (downloads ~3-4 GB from adsb.lol)
uv run python scripts/ingest_adsb.py 2024-04-08 --region north_america

# blind search
uv run python scripts/locate_recording.py my_recording.m4a --date 2024-04-08 \
    --region "37.0,38.5,-90.0,-88.5,north_america" --start "2024-04-08 14:00" --tz America/Chicago --window-min 10
```

Everything runs locally. Recordings never leave your machine; only flight data, terrain tiles,
weather and (optionally) place-name lookups are downloaded.

## Data sources

| what | source | licence |
|---|---|---|
| Historical ADS-B tracks | [adsb.lol](https://adsb.lol) daily `globe_history` archives | ODbL 1.0 |
| Aircraft type designators | [tar1090-db](https://github.com/wiedehopf/tar1090-db) (downloaded on first use) | see project |
| Terrain | Copernicus DEM GLO-90 | Copernicus licence |
| Geoid | EGM96 (via PROJ CDN) | public domain |
| Weather | [Open-Meteo](https://open-meteo.com) | CC BY 4.0 |

See [docs/DATA.md](docs/DATA.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Limitations

- Needs aircraft that broadcast ADS-B and a volunteer receiver nearby. Many light aircraft outside
  controlled airspace carry no ADS-B; low-altitude coverage varies by region.
- Needs aircraft close enough to hear: props within ~5 km, jets within ~3 km and low. 30-60 minutes
  of recording helps a lot.
- Device clocks are often off by tens of seconds; give an honest time window.
- Look-alike places exist (e.g. two spots along one approach corridor). The confidence margin and the
  candidate list show when the answer is ambiguous.

## Repository layout

```
src/dopplerdex/
  physics/   frames, trajectories, retarded-time Doppler solver, absorption, ground reflection
  sim/       physically based flyover renderer (synthetic test audio), sources, noise, codecs
  audio/     spectrogram front-end, tonal ridge tracking + Doppler fits, multipath (cepstrum) events
  adsb/      adsb.lol archive ingest, track store, smoothing
  env/       terrain, geoid, weather
  search/    coarse search, single-event search, end-to-end locate()
  scoring/   ridge matching, whole-recording consistency, explained events, refinement
  eval/      synthetic and real-track benchmark scenarios
  web/       FastAPI app + static UI (Leaflet map, Doppler lab)
bench/       benchmarks and comparisons with other open-source projects
scripts/     command-line tools
tests/       unit, property and integration tests
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first; in particular,
**never commit audio, video or private location data**. A hygiene check enforces this.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE). Third-party components and data: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

The method was popularised by colsto's video *"I discovered a sound that leaks your exact location"*.
Related open-source work: [hermes-aircraft-doppler-geolocation](https://github.com/mojomast/hermes-aircraft-doppler-geolocation)
and [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST). Ground-reflection delay events follow Lo & Ferguson (IEEE TAES 2002, 2003).
