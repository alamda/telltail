#!/usr/bin/env python3
"""Consolidated sweep: every method x standardization row of the README table.

One fit and one scoring pass per standardization; every window observable is
derived from the cached scored frame. Each stage runs in its own subprocess, so
peak memory is one stage, and writes rows_<stage>.json. Assembly checks that
every row shares one alert budget and one red-team denominator, then prints the
table.

    python src/sweep_observables.py --input data/lanl/derived/compromised_users.csv

Split is always the first red-team event, as in the published table.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

import detect
import surprisal as surp
from evaluate import (DAY_SECONDS, DEFAULT_BUDGETS, _ranked_by_day,
                      recall_at_budget)
from timesplit import FIRST_REDTEAM_TIME, split_frame, eval_window

# Scoring streams to disk; ranking uses detect.py's engine, so numbers reproduce
SINK_ENGINE = "streaming"
STANDARDIZATIONS = ("mad", "meanstd", "logz")
STAGES = ("novelty", "iforest", *STANDARDIZATIONS, "edges")

# Published row order; unlisted rows (extra temperatures) print last
ROW_ORDER = [
    ("Isolation Forest (baseline)", "-"),
    ("Novelty edges (baseline)", "-"),
    ("Free energy `G_T`, T=8", "mad"),
    ("Window variance", "mad"),
    ("Distinct-edge count", "mad"),
    ("Per-event z", "mad"),
    ("Per-event z", "meanstd"),
    ("Per-event z", "logz"),
    ("Window mean (default)", "mad"),
    ("Window mean", "meanstd"),
    ("Window mean", "logz"),
]


def _events(input_path, columns):
    return pl.scan_csv(input_path).select(columns)


def _redteam(events, split):
    """Recall denominator. Refit per stage; assembly checks the stages agree."""
    return int(eval_window(events, split)
               .select((pl.col("label") == 1).sum()).collect().item())


def _unstable_days(ranked, budgets):
    """Days whose cell could move if the engine reordered tied rows.

    A day counts when rows sharing the cutoff score straddle the budget and
    carry mixed labels, so which one is reviewed decides whether a red-team
    event is caught. All-benign or all-inside ties cannot move the count.
    """
    by_day = _ranked_by_day(ranked, "time", "score", DAY_SECONDS)
    unstable = {}
    for b in budgets:
        cutoff = (by_day.filter(pl.col("day_rank") == b - 1)
                  .select("day", pl.col("score").alias("cutoff_score")))
        tied = (by_day.join(cutoff, on="day")
                .filter(pl.col("score") == pl.col("cutoff_score"))
                .group_by("day")
                .agg(outside=(pl.col("day_rank") >= b).sum(),
                     labels=pl.col("label").n_unique()))
        unstable[str(b)] = tied.filter((pl.col("outside") > 0)
                                       & (pl.col("labels") > 1)).height
    return unstable


def _row(method, standardization, ranked, redteam, budgets):
    """Score one ranked frame into a table row."""
    table = recall_at_budget(ranked, redteam, budgets=budgets)
    keyed = {str(r["budget"]): r for r in table.to_dicts()}
    return {
        "method": method,
        "standardization": standardization,
        "n_scored": ranked.height,
        "redteam": redteam,
        "caught": {b: r["redteam_caught"] for b, r in keyed.items()},
        "alerts_used": {b: r["alerts_used"] for b, r in keyed.items()},
        "recall": {b: r["recall"] for b, r in keyed.items()},
        "unstable_days": _unstable_days(ranked, budgets),
    }


def _fit(input_path, method):
    """Count tables and per-user moments for `method`.

    Moments are refit per stage, never shared. The printed pooled scale makes
    that checkable: `standardize` floors every per-user scale at it and it is
    method-dependent (mad near 1.19, logz near 2.055).
    """
    events = _events(input_path, surp.NEEDED)
    split = FIRST_REDTEAM_TIME
    train, test = split_frame(events, split)
    ct = surp.count_tables(train)
    moments = surp.fit_moments(train, ct, method=method)
    print(f"[{method}] pooled center {moments.center:.4f}, "
          f"scale {moments.scale:.4f}")
    return events, split, test, ct, moments


def stage_baseline(model, input_path, budgets):
    """One baseline row: novelty or iforest, no standardization."""
    events = _events(input_path, detect.REQUIRED[model])
    split = FIRST_REDTEAM_TIME
    ranked = detect.SCORERS[model](events, split)
    label = ("Novelty edges (baseline)" if model == "novelty"
             else "Isolation Forest (baseline)")
    return [_row(label, "-", ranked, _redteam(events, split), budgets)]


def stage_standardization(method, input_path, budgets, window, temps, out_dir):
    """Every window observable for one standardization, from one scoring pass.

    mad also carries free energy and window variance, which the published
    table reports only there.
    """
    events, split, test, ct, moments = _fit(input_path, method)
    redteam = _redteam(events, split)
    scored = surp.standardize(surp.event_surprisal(test, ct), moments)

    # one expensive pass; every observable below reads this file
    cache = out_dir / f"scored_{method}.parquet"
    (scored.select(surp.USER, "time", "z", "label")
     .sink_parquet(cache, engine=SINK_ENGINE))
    base = pl.read_parquet(cache).lazy()
    print(f"[{method}] scored frame cached to {cache}")

    rows = [_row("Per-event z", method,
                 detect.rank_scored(base.with_columns(score=pl.col("z"))),
                 redteam, budgets)]

    windowed = surp.window_observable(base, window, None, "mean") \
        .rename({"z_win": "win_mean"})
    if method == "mad":
        windowed = surp.window_observable(windowed, window, None, "var") \
            .rename({"z_win": "win_var"})
    win = windowed.collect()

    mean_label = "Window mean (default)" if method == "mad" else "Window mean"
    rows.append(_row(mean_label, method,
                     detect.rank_scored(
                         win.lazy().with_columns(score=pl.col("win_mean"))),
                     redteam, budgets))

    if method == "mad":
        rows.append(_row("Window variance", method,
                         detect.rank_scored(
                             win.lazy().with_columns(score=pl.col("win_var"))),
                         redteam, budgets))
        del win
        for t in temps:
            free = surp.window_observable(base, window, t, None)
            rows.append(_row(f"Free energy `G_T`, T={t:g}", method,
                             detect.rank_scored(
                                 free.with_columns(score=pl.col("z_win"))),
                             redteam, budgets))
    return rows


def stage_edges(input_path, budgets, window):
    """Distinct-edge count. Its own stage: the rolling n_unique has unknown
    cost and needs the string edge keys other stages project away."""
    events, split, test, ct, moments = _fit(input_path, "mad")
    redteam = _redteam(events, split)
    scored = surp.standardize(surp.event_surprisal(test, ct), moments)

    windowed = surp.window_observable(scored, window, None, "edges")
    ranked = detect.rank_scored(windowed.with_columns(score=pl.col("z_win")))
    # the edges join drops or duplicates on a repeated (user, time)
    n_test = int(test.select(pl.len()).collect().item())
    if ranked.height != n_test:
        raise SystemExit(f"edges join changed row count: {n_test:,} test rows "
                         f"in, {ranked.height:,} out")
    return [_row("Distinct-edge count", "mad", ranked, redteam, budgets)]


def run_stage(stage, input_path, budgets, window, temps, out_dir):
    if stage in ("novelty", "iforest"):
        return stage_baseline(stage, input_path, budgets)
    if stage == "edges":
        return stage_edges(input_path, budgets, window)
    return stage_standardization(stage, input_path, budgets, window, temps,
                                 out_dir)


def _sort_key(row):
    key = (row["method"], row["standardization"])
    return ROW_ORDER.index(key) if key in ROW_ORDER else len(ROW_ORDER)


def assemble(out_dir, budgets):
    """Read every stage's rows back, check them against each other, print.

    Always every stage, not just the ones this run rebuilt, so a partial rerun
    cannot overwrite the table with a subset of its rows.
    """
    rows = []
    for stage in STAGES:
        path = out_dir / f"rows_{stage}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}, stage {stage} did not finish")
        rows += json.loads(path.read_text())
    rows.sort(key=_sort_key)

    problems = []
    redteam = {r["redteam"] for r in rows}
    if len(redteam) != 1:
        problems.append(f"red-team denominator differs across rows: {redteam}")
    for b in map(str, budgets):
        used = {r["alerts_used"][b] for r in rows}
        if len(used) != 1:
            # a differing row has lost or duplicated test events
            problems.append(f"alerts_used at B={b} differs across rows: {used}")

    header = ("| Method | Standardization | "
              + " | ".join(f"B={b}" for b in budgets) + " |")
    rule = "| --- | --- | " + " | ".join("---" for _ in budgets) + " |"
    lines = [header, rule]
    for r in rows:
        cells = " | ".join(str(r["caught"][str(b)]) for b in budgets)
        lines.append(f"| {r['method']} | {r['standardization']} | {cells} |")
    table = "\n".join(lines)

    n = rows[0]["redteam"]
    print(f"\nred-team events in test window: {n:,}")
    print("alerts_used: " + ", ".join(
        f"B={b} {rows[0]['alerts_used'][str(b)]:,}" for b in budgets))
    print(f"test events scored: {rows[0]['n_scored']:,}")
    print(f"\nred-team events caught out of {n}\n")
    print(table)

    unstable = {r["method"]: r["unstable_days"] for r in rows
                if any(r["unstable_days"].values())}
    if unstable:
        print("\ncells a reordering of tied rows could move, by test day:")
        for method, days in unstable.items():
            print(f"  {method}: "
                  + ", ".join(f"B={b} {d} days" for b, d in days.items() if d))
    else:
        print("\nno cell depends on the order of tied rows")

    (out_dir / "table.md").write_text(table + "\n")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwritten to {out_dir}/table.md and {out_dir}/rows.json")

    if problems:
        raise SystemExit("\nSWEEP INVALID, treat as a bug and not a result:\n"
                         + "\n".join("  " + p for p in problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input",
                        default="data/lanl/derived/compromised_users.csv",
                        help="labeled auth subset CSV")
    parser.add_argument("--out-dir", default="outputs/sweep",
                        help="stage results, scored-frame cache, final table")
    parser.add_argument("--budget", type=int, nargs="+",
                        default=list(DEFAULT_BUDGETS),
                        help="per-day alert budgets; "
                             f"default {list(DEFAULT_BUDGETS)}")
    parser.add_argument("--window", type=int, default=surp.WINDOW_SECONDS,
                        help="trailing window in seconds; "
                             f"default {surp.WINDOW_SECONDS}")
    parser.add_argument("--temps", type=float, nargs="+", default=[8.0],
                        help="free-energy temperatures; default 8, the "
                             "published value")
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(STAGES), help="stages to run; default all")
    parser.add_argument("--force", action="store_true",
                        help="rerun stages whose results already exist")
    parser.add_argument("--stage", choices=STAGES,
                        help="internal: run this one stage in this process")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage:
        rows = run_stage(args.stage, args.input, args.budget, args.window,
                         args.temps, out_dir)
        (out_dir / f"rows_{args.stage}.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        return

    for stage in args.stages:
        path = out_dir / f"rows_{stage}.json"
        if path.exists() and not args.force:
            print(f"[{stage}] skipped, {path} exists (--force to redo)")
            continue
        print(f"[{stage}] start")
        started = time.monotonic()
        subprocess.run(
            [sys.executable, __file__, "--stage", stage,
             "--input", args.input, "--out-dir", str(out_dir),
             "--budget", *map(str, args.budget), "--window", str(args.window),
             "--temps", *map(str, args.temps)], check=True)
        print(f"[{stage}] done in {time.monotonic() - started:.0f}s")

    assemble(out_dir, args.budget)


if __name__ == "__main__":
    main()
