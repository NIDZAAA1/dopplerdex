"""uv run python scripts/ingest_adsb.py 2026-09-08 [--region north_america] [--workers 8]"""

import argparse
import json
from pathlib import Path

from dopplerdex.adsb.ingest import REGIONS, ingest_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--region", default="north_america", choices=list(REGIONS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--raw", default="data/raw/adsblol")
    ap.add_argument("--out", default="data/processed/adsb")
    args = ap.parse_args()
    parts = sorted(Path(args.raw, args.date).glob("*.tar.a?"))
    if not parts:
        raise SystemExit(f"no archive parts in {args.raw}/{args.date}")
    out = Path(args.out) / f"region={args.region}" / f"date={args.date}"
    stats = ingest_day(parts, out, REGIONS[args.region], workers=args.workers)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
