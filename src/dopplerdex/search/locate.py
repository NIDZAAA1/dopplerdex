"""End-to-end blind localisation.

audio -> Doppler events -> for each event pair: coarse search (whole region, whole time range) with
ridge-match scoring -> best flight-pair hypotheses -> whole-recording loudness consistency (all
traffic) -> continuous refinement of the winners -> ranked locations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from dopplerdex.audio.events import DetectConfig, DetectedEvent, detect_events
from dopplerdex.audio.loud_events import LoudEvent, detect_loud_events
from dopplerdex.audio.multipath import MultipathEvent, detect_multipath
from dopplerdex.scoring.consistency import explained_events, loudness_from_passes, predicted_passes_at
from dopplerdex.scoring.refine import refine_hypothesis
from dopplerdex.scoring.ridge_match import RidgeSet
from dopplerdex.scoring.spectro_consistency import spectro_consistency
from dopplerdex.search.coarse import EventConstraint
from dopplerdex.search.loudness import observed_level
from dopplerdex.search.pipeline import SearchRegion, coarse_search, coarse_single


@dataclass
class LocateResult:
    ranked: pl.DataFrame  # hypotheses after loudness re-ranking (best first), with refined positions for the top ones
    events: list[DetectedEvent]
    pairs_tried: list[tuple[int, int]]
    stats: dict = field(default_factory=dict)

    @property
    def best(self) -> dict | None:
        return self.ranked.row(0, named=True) if len(self.ranked) else None


def choose_pairs(events: list[DetectedEvent], max_pairs: int = 3, min_sep_s: float = 60.0, pool: int = 6,
                 max_uses: int = 2) -> list[tuple[int, int]]:  # fmt: skip
    """Pairs from the strongest events, spread over several anchors.

    Anchoring every pair on one event fails whenever that aircraft is absent from ADS-B (no transponder,
    no receiver coverage at low altitude): then no pair contains the truth. So candidate pairs come from
    the top `pool` events (tier A first, then evidence), are taken best-first by combined rank, and no
    event is used in more than `max_uses` pairs."""
    air = [i for i in range(len(events)) if events[i].tier != "G"]  # never search with road traffic
    if len(air) < 2:
        return []
    order = sorted(air, key=lambda i: (events[i].tier != "A", -events[i].evidence))[:pool]
    rank = {e: k for k, e in enumerate(order)}
    cand = [(a, b) for k, a in enumerate(order) for b in order[k + 1 :]
            if abs(events[a].fit.t_cpa_heard - events[b].fit.t_cpa_heard) >= min_sep_s]  # fmt: skip
    cand.sort(key=lambda p: (rank[p[0]] + rank[p[1]], max(rank[p[0]], rank[p[1]])))
    uses: dict[int, int] = {}
    out = []
    for a, b in cand:
        if uses.get(a, 0) >= max_uses or uses.get(b, 0) >= max_uses:
            continue
        out.append((a, b) if events[a].fit.t_cpa_heard < events[b].fit.t_cpa_heard else (b, a))
        uses[a], uses[b] = uses.get(a, 0) + 1, uses.get(b, 0) + 1
        if len(out) == max_pairs:
            break
    return out


def locate(
    x: np.ndarray,
    fs: float,
    t0_lo: float,
    t0_hi: float,
    c: float,
    region: SearchRegion = SearchRegion(),
    max_pairs: int = 3,
    pairs_per_search: int = 60,
    pairs_per_loud_search: int = 300,
    n_refine: int = 3,
    events: list[DetectedEvent] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    verbose: bool = False,
    progress=None,  # optional callable(fraction in [0, 1], message)
    ranking: str = "loud",  # "loud" | "spectro" | "combined" | "ridge"
    single_flights: int = 15,
    single_cells_per_flight: int = 30,
) -> LocateResult:
    """Blind localisation. Default ranking = whole-recording loudness consistency: on the reference
    challenge and the synthetic benchmark it ranked the truth #1 where the combined/spectrogram scores
    did not (spectrogram evidence is inflated for hypotheses that predict many tonal passes)."""
    t_start = time.time()

    def report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(float(frac), msg)

    report(0.0, "detecting aircraft events")
    fm = None
    if events is None:
        events, fm = detect_events(x, fs, DetectConfig(c=c))
    if fm is None:
        _, fm = detect_events(x, fs, DetectConfig(c=c))
    air = [i for i in range(len(events)) if events[i].tier != "G"]
    if len(air) >= 6:
        max_pairs = max(max_pairs, 5)  # busy recording: more independent pairs (cheap next to a wrong anchor)
    pairs = pairs if pairs is not None else choose_pairs(events, max_pairs)
    # Ground-reflection (multipath) passes: mostly jets without a usable tonal line. A pass also heard as a
    # tonal curve (within 15 s) is the same aircraft: the tonal event wins.
    report(0.02, "detecting ground-reflection passes")
    mp_events: list[MultipathEvent] = [
        m for m in detect_multipath(x, fs) if all(abs(m.t_cpa - events[i].fit.t_cpa_heard) > 15.0 for i in air)
    ]
    jobs: list[tuple[str, int, int]] = [("tonal", ia, ib) for ia, ib in pairs]
    anchor = min(air, key=lambda i: (events[i].tier != "A", -events[i].evidence)) if air else None
    if len(jobs) < max_pairs and anchor is not None:
        # strongest tonal curve + ground-reflection passes (their T = S0 / v gives range from the ADS-B speed)
        ta = events[anchor].fit.t_cpa_heard
        jobs += [("mp", anchor, j) for j, m in enumerate(mp_events) if abs(m.t_cpa - ta) > 60][: max_pairs - len(jobs)]
    elif anchor is None and len(mp_events) >= 2:
        # no tonal curve at all: pair the strongest ground-reflection pass with the next well-separated ones
        jobs += [("mpmp", 0, j) for j in range(1, len(mp_events)) if abs(mp_events[j].t_cpa - mp_events[0].t_cpa) > 60][
            :max_pairs
        ]
    # Still too few: pair the strongest tonal event with loud broadband passes (timing-only partners).
    loud_events: list[LoudEvent] = []
    if len(jobs) < max_pairs and anchor is not None:
        ta = events[anchor].fit.t_cpa_heard
        loud_events = [e for e in detect_loud_events(x, fs) if abs(e.t_peak - ta) > 60][: max_pairs - len(jobs)]
    jobs += [("loud", anchor, k) for k in range(len(loud_events))]
    # Only one usable aircraft curve and nothing to pair it with geometrically: single-event search
    if not any(k in ("tonal", "mp", "mpmp") for k, _, _ in jobs) and anchor is not None:
        jobs.append(("single", anchor, -1))
    tabs = []
    for n_job, (kind, ia, ib) in enumerate(jobs):
        lo, hi = 0.05 + 0.75 * n_job / max(len(jobs), 1), 0.05 + 0.75 * (n_job + 1) / max(len(jobs), 1)
        label = f"search {n_job + 1}/{len(jobs)} ({kind} pair)"
        report(lo, label)
        if kind == "mpmp":
            ca, ra = _mp_constraint(mp_events[ia]), None
        else:
            ca, ra = EventConstraint.from_fit(events[ia].fit), RidgeSet.from_event(events[ia])
        if kind == "single":
            res = coarse_single(ca, t0_lo, t0_hi, c, region, ridges=ra, cell_m=500.0, verbose=verbose,
                                progress=lambda f, m, lo=lo, hi=hi, label=label: report(lo + (hi - lo) * f, f"{label}: {m}"))  # fmt: skip
            h = res.hypotheses
            if h.is_empty() or "score" not in h.columns:
                continue
            h = _single_candidates(h, single_flights, single_cells_per_flight)
            h = h.join(res.flights.select(["id", "icao", "type"]), left_on="fid_a", right_on="id", how="left")
            h = h.with_columns(pl.lit(None, pl.Utf8).alias("icao_b"), pl.lit(None, pl.Utf8).alias("type_b"))
            tabs.append(
                h.with_columns(
                    pl.lit(ia).alias("event_a"), pl.lit(-1).alias("event_b"), pl.lit(kind).alias("pair_kind")
                )
            )
            continue
        if kind == "tonal":
            cb, rb = EventConstraint.from_fit(events[ib].fit), RidgeSet.from_event(events[ib])
        elif kind in ("mp", "mpmp"):
            cb, rb = _mp_constraint(mp_events[ib]), (RidgeSet.empty() if ra is not None else None)
        else:
            le = loud_events[ib]
            cb = EventConstraint.from_values(150.0, 60.0, 3500.0, 1500.0, le.t_peak + 3.0, 8.0, r_margin=500.0, classes=(3, 4),
                                             label=f"loud@{le.t_peak:.0f}")  # fmt: skip
            rb = RidgeSet.empty()
        res = coarse_search(ca, cb, t0_lo, t0_hi, c, region, ridges_a=ra, ridges_b=rb, verbose=verbose,
                            progress=lambda f, m, lo=lo, hi=hi, label=label: report(lo + (hi - lo) * f, f"{label}: {m}"))  # fmt: skip
        h = res.hypotheses
        if not h.is_empty() and kind == "mpmp":
            h = h.with_columns(
                pl.col("dt0").abs().alias("score"), pl.col("t0").alias("t0_fit")
            )  # no ridges: timing agreement
        if h.is_empty() or "score" not in h.columns:
            continue
        # timing-only (loud) partners leave the ridge score weakly selective: send more into consistency
        keep = pairs_per_search if kind == "tonal" else pairs_per_loud_search  # mp / mpmp / loud: wider
        h = h.sort("score").unique(subset=["fid_a", "fid_b"], keep="first").sort("score").head(keep)
        h = h.join(res.flights.select(["id", "icao", "type"]), left_on="fid_a", right_on="id", how="left")
        h = h.join(res.flights.select(["id", "icao", "type"]), left_on="fid_b", right_on="id", how="left", suffix="_b")
        tabs.append(
            h.with_columns(pl.lit(ia).alias("event_a"), pl.lit(ib).alias("event_b"), pl.lit(kind).alias("pair_kind"))
        )
    if not tabs:
        return LocateResult(pl.DataFrame(), events, pairs, {"seconds": time.time() - t_start, "n_events": len(events)})
    cand = pl.concat(tabs, how="diagonal_relaxed")
    if not any(k in ("tonal", "mpmp") for k, _, _ in jobs):
        # Without a tonal pair the curve fit is the only geometric evidence: drop flights that reproduce the
        # one detected curve much worse than the best flight does (same relative rule as within a flight).
        cand = cand.filter(pl.col("score") <= 1.5 * cand["score"].min() + 5.0)

    # whole-recording consistency with ALL predicted traffic: loudness envelope (incl. jets) and
    # Doppler-compensated tonal evidence for every predicted prop/helicopter pass (soft silence filter)
    t_rec, lv = observed_level(x, fs, 300.0, min(3000.0, 0.45 * fs))
    dur = float(len(x) / fs)
    loud, spec, n_conf, n_expl, expl = [], [], [], [], []
    n_cand = len(cand)
    for k_c, r in enumerate(cand.iter_rows(named=True)):
        if k_c % 10 == 0:
            report(
                0.80 + 0.15 * k_c / max(n_cand, 1), f"checking candidates against the whole recording ({k_c}/{n_cand})"
            )
        ps = predicted_passes_at(r["lat"], r["lon"], r["t0_fit"], dur, c, region=region.adsb_region)
        loud.append(loudness_from_passes(t_rec, lv, ps, r["t0_fit"]))
        sc, evs = spectro_consistency(fm, r["t0_fit"], ps, dur)
        spec.append(sc)
        n_conf.append(sum(1 for e in evs if e.evidence > 4.0))
        _, ex = explained_events(ps, events, r["t0_fit"], mp_events=mp_events)
        # tier-A curves and ground-reflection passes count 1, weak tonal fragments (tier B) count half
        n_expl.append(float(sum(0.5 if k == "t" and events[i].tier == "B" else 1.0 for k, i, _, _ in ex)))
        expl.append(ex)

    def rz(v):
        v = np.asarray(v, dtype=float)
        mad = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9
        return (v - np.median(v)) / mad

    combined = rz(loud) + rz(spec)
    # Weak prior on the stated start time: the true start is likelier near the stated time than at the
    # window edges. At most 0.03 of correlation at the edge, so it only breaks near-ties (e.g. positions
    # along the strip of a single event, which the recording cannot tell apart).
    t_mid, half = 0.5 * (t0_lo + t0_hi), max(0.5 * (t0_hi - t0_lo), 1.0)
    prior = -0.03 * ((cand["t0_fit"].to_numpy() - t_mid) / half) ** 2
    loud = list(np.asarray(loud) + prior)
    cand = cand.with_columns(pl.Series("loud_corr", loud), pl.Series("spectro", spec), pl.Series("n_confirmed", n_conf),
                             pl.Series("combined", combined), pl.Series("n_explained", n_expl),
                             pl.Series("explained", [json.dumps(e) for e in expl]))  # fmt: skip
    # "single": no two tonal curves agree on timing (one curve, possibly with loud-pass partners)
    mode = "pairs" if any(k in ("tonal", "mp", "mpmp") for k, _, _ in jobs) else "single"
    if mode == "single" and ranking == "loud":
        # No second curve pins the timing, so loudness alone barely separates positions; the tonal
        # evidence of every predicted prop pass (incl. the matched one) is needed as well.
        ranking = "combined"
        cand = cand.with_columns(
            pl.Series("combined", combined + 20.0 * prior)
        )  # same start-time prior, z-score scale (<= 0.6)
    key, desc = {"loud": ("loud_corr", True), "spectro": ("spectro", True), "combined": ("combined", True),
                 "ridge": ("score", False)}[ranking]  # fmt: skip
    # Primary: how many detected curves the hypothesis explains (decoys that mimic two curves rarely
    # explain the rest); within the same count, the whole-recording consistency score decides.
    cand = cand.sort(["n_explained", key], descending=[True, desc])
    corr = list(cand[key]) if desc else [-v for v in cand[key]]

    if len(cand) > 1 and cand["pair_kind"][0] == "single":
        cand = _strip_centre(cand, key)
    # continuous refinement of the winners
    report(0.95, "refining the best candidates")
    ref_rows = []
    for k, r in enumerate(cand.head(n_refine).iter_rows(named=True)):
        kind = r.get("pair_kind", "tonal")
        ev_a = None if kind == "mpmp" else events[r["event_a"]]
        icao_a = None if kind == "mpmp" else r["icao"]
        icao_b, ev_b = (r["icao_b"], events[r["event_b"]]) if kind == "tonal" else (None, None)
        used_t = set() if kind == "mpmp" else {r["event_a"]} | ({r["event_b"]} if kind == "tonal" else set())
        used_m = {r["event_b"]} if kind == "mp" else ({r["event_a"], r["event_b"]} if kind == "mpmp" else set())
        ex = json.loads(r.get("explained") or "[]")
        # every further well-constrained curve this hypothesis explains joins the refinement (one shared
        # clock): tier-A tonal curves and ground-reflection passes; weak tier-B fragments only count for ranking
        extra = [(icao, events[i]) for kd, i, icao, _ in ex if kd == "t" and i not in used_t and events[i].tier == "A"]
        mp_extra = [(icao, mp_events[i]) for kd, i, icao, _ in ex if kd == "m" and i not in used_m]
        if kind == "mp":
            mp_extra.append((r["icao_b"], mp_events[r["event_b"]]))
        elif kind == "mpmp":
            mp_extra = [(r["icao"], mp_events[r["event_a"]]), (r["icao_b"], mp_events[r["event_b"]])] + mp_extra
        if kind == "loud" and not (extra or mp_extra):
            rf = None  # one curve + a timing-only partner: keep the consistency-checked coarse position
        else:
            rf = refine_hypothesis(r["lat"], r["lon"], r["t0_fit"], icao_a, icao_b, ev_a, ev_b,
                                   region=region.adsb_region, extra=extra, mp_extra=mp_extra)  # fmt: skip
        n_used = (ev_a is not None) + (ev_b is not None) + len(extra) + len(mp_extra)
        ref_rows.append({"rank": k + 1, "lat_ref": rf.lat if rf else r["lat"], "lon_ref": rf.lon if rf else r["lon"],
                         "t0_ref": rf.t0 if rf else r["t0_fit"], "score_ref": rf.score if rf else None,
                         "n_refined": n_used if rf else 0})  # fmt: skip
    ranked = cand.with_row_index("rank", offset=1).join(pl.DataFrame(ref_rows), on="rank", how="left")
    margin = float(corr[0] - corr[1]) if len(corr) > 1 else None  # gap between #1 and #2 in the ranking score
    return LocateResult(ranked, events, pairs, {"seconds": time.time() - t_start, "n_events": len(events), "n_loud": len(loud_events), "ranking": ranking,
                                                 "loud_margin": margin, "mode": mode,
                                                 "n_multipath": len(mp_events)})  # fmt: skip


def _mp_constraint(m: MultipathEvent) -> EventConstraint:
    """Search constraint of a ground-reflection pass: heard CPA time, and range = T x speed (T = S0 / v from
    the delay curve), so conditioning on each aircraft's ADS-B speed pins its CPA range."""
    return EventConstraint.from_values(150.0, 80.0, m.T * 150.0, m.T * 80.0, m.t_cpa, 2.0, rho_vr=0.99, r_margin=300.0,
                                       label=f"multipath@{m.t_cpa:.0f}")  # fmt: skip


def _strip_centre(cand: pl.DataFrame, key: str, radius_m: float = 4000.0, temp: float = 1.0) -> pl.DataFrame:
    """Replace the winner's position by the evidence-weighted centre of its strip segment.

    Along one flight's strip the recording often cannot tell positions apart (moving along the track
    is equivalent to shifting the clock). The weighted mean of the winner's flight candidates within
    radius_m on the same side minimises the expected error instead of picking an arbitrary cell.
    Mirror-side candidates are ~2x the CPA ground offset away, beyond radius_m for typical ranges; the
    radius is capped below that offset so the two sides are never averaged."""
    top = cand.row(0, named=True)
    same = cand.filter(pl.col("fid_a") == top["fid_a"])
    lat0, lon0 = top["lat"], top["lon"]
    coslat = np.cos(np.radians(lat0))
    de = (same["lon"].to_numpy() - lon0) * 111320.0 * coslat
    dn = (same["lat"].to_numpy() - lat0) * 111320.0
    h = float(np.sqrt(max(top["r_a"] ** 2 - 1500.0**2, 250_000.0)))  # rough ground offset of the strip
    near = np.hypot(de, dn) < min(radius_m, 1.2 * h)
    v = same[key].to_numpy()[near]
    w = np.exp((v - v.max()) / temp)
    w /= w.sum()
    lat_c = float(np.sum(w * same["lat"].to_numpy()[near]))
    lon_c = float(np.sum(w * same["lon"].to_numpy()[near]))
    t0_c = float(np.sum(w * same["t0_fit"].to_numpy()[near]))
    first = cand.head(1).with_columns(
        pl.lit(lat_c).alias("lat"), pl.lit(lon_c).alias("lon"), pl.lit(t0_c).alias("t0_fit")
    )
    return pl.concat([first, cand.slice(1)], how="vertical_relaxed")


def _single_candidates(h: pl.DataFrame, n_flights: int, cells_per_flight: int) -> pl.DataFrame:
    """Best-matching flights, and for each up to cells_per_flight cells spread along both strips
    (cells whose curve fit is within 1.5x + 5 cents of that flight's best), ordered by start time so the
    subsample covers the whole along-track extent rather than clustering."""
    h = h.filter(pl.col("score") < 1e8)
    best = h.group_by("fid_a").agg(pl.col("score").min().alias("best")).sort("best").head(n_flights)
    h = h.join(best, on="fid_a").filter(pl.col("score") <= 1.5 * pl.col("best") + 5.0)
    out = []
    for g in h.partition_by("fid_a"):
        g = g.sort("t0_fit")
        idx = np.unique(np.linspace(0, len(g) - 1, min(cells_per_flight, len(g))).round().astype(int))
        out.append(g[idx])
    return pl.concat(out).drop("best").sort("score") if out else h
