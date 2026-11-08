#!/bin/zsh
# hermes on the eclipse clips through its own harness (published flags), cache warmed from the same adsb.lol day
cd "$(dirname "$0")/../.."
H=reference/hermes-aircraft-doppler-geolocation
export DOPPLER_GEO_CACHE=$PWD/data/cache/hermes
PARTS=$(python3 -c "import glob,json;print(json.dumps(sorted(glob.glob('data/raw/adsblol/2024-04-08/*.tar.a?'))))")
python3 -c "
import json;m=json.load(open('bench/eclipse/manifest_hermes.json'))
print('\n'.join(f\"{e['truth_lat']} {e['truth_lon']}\" for e in m))" | sort -u | while read la lo; do
  reference/_venv_hermes/bin/python -c "
import datetime as dt, json
from dopplergeo.adsb import fetch_tracks
d=dt.date(2024,4,8); s=dt.datetime(2024,4,8,tzinfo=dt.UTC).timestamp()
fetch_tracks(d,($la-0.6,$lo-0.9,$la+0.6,$lo+0.9),s,s+86400,tar_parts=[p for p in json.loads('$PARTS')])
print('warmed $la $lo')"
done
(cd $H && ../_venv_hermes/bin/python scripts/eval_batch.py $OLDPWD/bench/eclipse/manifest_hermes.json --out $OLDPWD/data/cache/hermes_eval/eclipse_tol30 --tol 30 --jobs 2 --det broadband=true --set temper_extra=4)
