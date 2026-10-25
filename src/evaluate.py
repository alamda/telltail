#!/usr/bin/env python3
"""Recall at a fixed alert budget for the baseline detectors.

Primary metric: per-day event review budget B. 
recall@B = fraction of red team events that fall in some day's top-B.

--verbose: global top-N (no per-day cap), sanity check

Input is a baseline's scored alerts (time, score, label, ...).
"""
import polars as pl

DAY_SECONDS = 86400
DEFAULT_BUDGETS = (10, 25, 50, 100)


def _ranked_by_day(scored, time_col, score_col, day_seconds):
    """Add `day` and `day_rank` (0-based, highest score first, earliest event first on ties)."""
    return (scored
            .with_columns((pl.col(time_col) // day_seconds).alias("day"))
            .sort(["day", score_col, time_col], descending=[False, True, False])
            .with_columns(pl.int_range(pl.len()).over("day").alias("day_rank")))


def recall_at_budget(scored, total_redteam, budgets=DEFAULT_BUDGETS, *,
                     time_col="time", score_col="score", label_col="label",
                     day_seconds=DAY_SECONDS):
    """Per-day recall@budget table.

    scored: alert rows with time, score, label columns, one row per scored event.
    total_redteam: test-window red team count

    Returns frame: budget, alerts_used, redteam_caught, false_positives,
    recall, precision. false_positives = alerts_used - redteam_caught (benign
    events in the top-B); precision = redteam_caught / alerts_used.
    """
    ranked = _ranked_by_day(scored, time_col, score_col, day_seconds)
    rows = []
    for b in budgets:
        within = ranked.filter(pl.col("day_rank") < b)
        used = within.height
        caught = int(within.filter(pl.col(label_col) == 1).height)
        rows.append({"budget": b, "alerts_used": used,
                     "redteam_caught": caught,
                     "false_positives": used - caught,
                     "recall": caught / total_redteam if total_redteam else 0.0,
                     "precision": caught / used if used else 0.0})
    return pl.DataFrame(rows)


def recall_at_budget_global(scored, total_redteam, budgets=DEFAULT_BUDGETS, *,
                            time_col="time", score_col="score", label_col="label",
                            day_seconds=DAY_SECONDS):
    """Global top-N sanity check

    Returns frame: budget, total_alerts, redteam_caught, false_positives,
    recall, precision.
    """
    n_days = scored.select((pl.col(time_col) // day_seconds).n_unique()).item()
    ranked = scored.sort([score_col, time_col], descending=[True, False])
    rows = []
    for b in budgets:
        n = b * n_days
        within = ranked.head(n)
        used = within.height
        caught = int(within.filter(pl.col(label_col) == 1).height)
        rows.append({"budget": b, "total_alerts": n, "redteam_caught": caught,
                     "false_positives": used - caught,
                     "recall": caught / total_redteam if total_redteam else 0.0,
                     "precision": caught / used if used else 0.0})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    from timesplit import (FIRST_REDTEAM_TIME, split_time_from_fraction,
                           eval_window)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("alerts", help="scored alerts CSV (time, score, label)")
    parser.add_argument("--events", required=True,
                        help="labeled auth subset CSV, ground-truth source for "
                             "test-window red team count")
    parser.add_argument("--split-fraction", type=float, default=None,
                        help="train/test split at this fraction of the time range "
                             "(0-1); default is first red team event timestamp")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=list(DEFAULT_BUDGETS),
                        help="per-day alert budgets to report")
    parser.add_argument("--verbose", action="store_true",
                        help="also print global top-N sanity check")
    args = parser.parse_args()

    # Project only what recall_at_budget needs
    # Eager-reading every column risks avoidable OOM
    alerts = pl.scan_csv(args.alerts).select("time", "score", "label").collect()

    # Denominator is test-window red team count from events file
    events = pl.scan_csv(args.events)
    if args.split_fraction is None:
        split = FIRST_REDTEAM_TIME
    else:
        split = split_time_from_fraction(events, args.split_fraction)
    total = int(eval_window(events, split)
                .select((pl.col("label") == 1).sum()).collect().item())

    print(f"per-day recall@budget (denominator: {total} red team)")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(recall_at_budget(alerts, total, budgets=args.budgets))
    if args.verbose:
        print("\nglobal top-N sanity check (same pooled budget, no per-day cap)")
        with pl.Config(tbl_hide_dataframe_shape=True):
            print(recall_at_budget_global(alerts, total, budgets=args.budgets))
