"""Summarise bench/compare/results.jsonl into a Markdown table (bench/compare/SUMMARY.md)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"ours": "dopplerdex (ours)", "hermes": "hermes / dopplergeo", "dg": "DOPPLERGEIST"}


def main() -> None:
    rows = [json.loads(line) for line in (HERE / "results.jsonl").read_text().splitlines() if line.strip()]
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    cases = sorted({r["case"] for r in rows})
    out = ["# Head-to-head comparison", "",
           "Protocol (hermes' published local protocol): 15 km search radius around a centre 5 km from the truth "
           "(bearing from the case id), recording start known to +-30 s (+-30 min where the recordist gave only an approximate time); "
           "identical inputs; truth used only for scoring. "
           "Patches to other systems: see PATCHES.md.", "",
           f"Cases: {len(cases)}", "",
           "| system | cases | answered | < 100 m | < 1 km | < 3 km | median error (answered) | truth within 1.5 km in top-20 | median run time |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in ("ours", "hermes", "dg"):
        rs = by_sys.get(s, [])
        if not rs:
            continue
        ans = [r for r in rs if r.get("answered")]
        errs = [r["error_m"] for r in ans if r.get("error_m") is not None]
        within = [r for r in ans if r.get("rank_within_1500m")]
        med = f"{statistics.median(errs) / 1000:.2f} km" if errs else "-"
        out.append(f"| {NAMES[s]} | {len(rs)} | {len(ans)} | {sum(e < 100 for e in errs)} | {sum(e < 1000 for e in errs)} | "
                   f"{sum(e < 3000 for e in errs)} | {med} | {len(within)} | {statistics.median(r['seconds'] for r in rs):.0f} s |")  # fmt: skip
    out += [
        "",
        "## Per case",
        "",
        "| case | " + " | ".join(NAMES[s] for s in ("ours", "hermes", "dg")) + " |",
        "|---|---|---|---|",
    ]
    for c in cases:
        cells = []
        for s in ("ours", "hermes", "dg"):
            r = next((x for x in rows if x["case"] == c and x["system"] == s), None)
            if r is None:
                cells.append("not run")
            elif r.get("answered"):
                cells.append(
                    f"{r['error_m'] / 1000:.2f} km"
                    + (f" (rank {r['rank_within_1500m']})" if r.get("rank_within_1500m") else "")
                )
            else:
                cells.append("no answer: " + str(r.get("error") or "")[:60].replace("|", "/"))
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    (HERE / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
