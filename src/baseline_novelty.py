#!/usr/bin/env python3
"""Baseline A: flag previously unseen (src_comp, dst_comp, auth_type) auth edges.

The detector is an anti-join: test edges absent from the train edge set.

Alerts carry a tier from the coarser (dst_comp, auth_type) edge:
  tier 0: (dst_comp, auth_type) also unseen in train (host meets a new protocol)
  tier 1: only the src_comp is new for a known (dst_comp, auth_type)
Pair-novel events are a subset of triple-novel ones, so tier 0 is the narrower
set and ranks first.

Schema (from make_subsets.py):
  time, src_user, dst_user, src_comp, dst_comp, auth_type,
  logon_type, auth_orientation, outcome, label
"""
import polars as pl

from timesplit import train_window, eval_window

EDGE_KEYS = ("src_comp", "dst_comp", "auth_type")
PAIR_KEYS = ("dst_comp", "auth_type")

# Columns this baseline reads. Project scan_csv to these so the wide auth schema
# (dst_user, logon_type, auth_orientation, outcome) never loads.
NEEDED = ("time", "src_user", "src_comp", "dst_comp", "auth_type", "label")


def train_edges(frame, split, keys=EDGE_KEYS):
    """Unique edges (over keys) seen strictly before split."""
    return (train_window(frame, split)
            .select(keys)
            .unique())


def score_events(frame, edges, split, keys=EDGE_KEYS):
    """Test rows (time >= split) whose edge (over keys) is absent from edges.

    An edge with no train history alerts on every occurrence.
    """
    return (eval_window(frame, split)
            .join(edges, on=list(keys), how="anti"))


def assign_tier(alerts, seen_pairs, keys=PAIR_KEYS):
    """Tag each alert with a tier from the coarser (dst_comp, auth_type) edge.

    tier 0: pair unseen in train (also novel); tier 1: pair already seen.
    """
    seen = seen_pairs.with_columns(pl.lit(1, dtype=pl.Int64).alias("tier"))
    return (alerts.join(seen, on=list(keys), how="left")
            .with_columns(pl.col("tier").fill_null(0)))


def dest_popularity(frame, split, user_col="src_user", dest_col="dst_comp"):
    """Distinct source users per destination in the train window.

    New connection to unpopular host more sus than one to popular host
    """
    return (train_window(frame, split)
            .group_by(dest_col)
            .agg(pl.col(user_col).n_unique().alias("train_dest_users")))


def rank_alerts(alerts, popularity, dest_col="dst_comp"):
    """Order alerts by tier, then destination rarity, ties broken by time.

    tier 0 (pair also novel) outranks tier 1 (only src is new).
    Within a tier, score = 1 / (1 + distinct train users of the destination).

    unseen host: score = 1
    popular host: score -> 0
    """
    return (alerts.join(popularity, on=dest_col, how="left")
            .with_columns(pl.col("train_dest_users").fill_null(0))
            .with_columns((1 / (1 + pl.col("train_dest_users"))).alias("score"))
            .sort(["tier", "score", "time"], descending=[False, True, False]))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from evaluate import recall_at_budget
    from timesplit import FIRST_REDTEAM_TIME, split_time_from_fraction

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="labeled auth subset CSV")
    parser.add_argument("--split-fraction", type=float, default=None,
                        help="train/test split at this fraction of the time range "
                             "(0-1); default is the first red team event")
    args = parser.parse_args()

    events = pl.scan_csv(args.input).select(NEEDED)
    if args.split_fraction is None:
        split = FIRST_REDTEAM_TIME
    else:
        split = split_time_from_fraction(events, args.split_fraction)

    seen_edges = train_edges(events, split)
    novel_edges = score_events(events, seen_edges, split)
    seen_pairs = train_edges(events, split, keys=PAIR_KEYS)
    tiered = assign_tier(novel_edges, seen_pairs)
    dest_user_counts = dest_popularity(events, split)
    ranked_alerts = rank_alerts(tiered, dest_user_counts).collect()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "novelty_alerts.csv"
    ranked_alerts.write_csv(out_path)

    eval_events = eval_window(events, split)
    eval_totals = eval_events.select(rows=pl.len(), redteam=(pl.col("label") == 1).sum()).collect()
    rows, redteam = eval_totals["rows"][0], eval_totals["redteam"][0]
    caught = ranked_alerts.filter(pl.col("label") == 1).height
    recall = caught / redteam if redteam else 0.0

    print(f"test {rows} rows; {ranked_alerts.height} alerts; "
          f"red team caught {caught}/{redteam} ({recall:.1%}) at unlimited budget")
    print(f"ranked alerts written to {out_path}")

    # tier 0 alone is the (dst_comp, auth_type) pair detector
    print("\nalerts / red team caught by tier")
    tier_stats = (ranked_alerts.group_by("tier")
                  .agg(alerts=pl.len(), caught=(pl.col("label") == 1).sum())
                  .sort("tier"))
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(tier_stats)

    # per-day budget is an analyst-facing metric
    print(f"\nper-day recall@budget (denominator: {redteam} red team)")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(recall_at_budget(ranked_alerts, int(redteam)))

    print("\ntop alerts:")
    top_alerts = ranked_alerts.head(10).select("time", "src_comp", "dst_comp",
                                               "auth_type", "tier", "score", "label")
    with pl.Config(tbl_rows=10, tbl_hide_dataframe_shape=True):
        print(top_alerts)
