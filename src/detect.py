#!/usr/bin/env python3
"""Unified detect CLI: score an auth subset with one method, report recall@budget.

    detect --input <csv> --model {novelty,iforest,surprisal,surprisal-mean,surprisal-var,surprisal-edges,all} --budget <B...>

One entrypoint over six detectors. Each model returns a ranked frame with
time, score, label columns. recall@budget is then the same metric for all of
them (evaluate.recall_at_budget). The denominator is the test-window red team
count read from the input itself.

--method sets the per-user standardization for the surprisal family
(surprisal.fit_moments).

--model all runs each detector in its own subprocess so peak memory stays one
model, not the sum (else the 17M-row iforest matrix stacks and OOMs).
"""
import polars as pl

import baseline_iforest as iforest
import baseline_novelty as novelty
import surprisal as surp
from evaluate import DEFAULT_BUDGETS, recall_at_budget
from timesplit import (FIRST_REDTEAM_TIME, split_time_from_fraction,
                       split_frame, eval_window)


def score_novelty(events, split, **_opts):
    """Rank test edges absent from the train edge set (baseline A)."""
    seen_edges = novelty.train_edges(events, split)
    novel_edges = novelty.score_events(events, seen_edges, split)
    seen_pairs = novelty.train_edges(events, split, keys=novelty.PAIR_KEYS)
    tiered = novelty.assign_tier(novel_edges, seen_pairs)
    dest_user_counts = novelty.dest_popularity(events, split)
    return novelty.rank_alerts(tiered, dest_user_counts).collect()


def score_iforest(events, split, **_opts):
    """Fit Isolation Forest on train, score test window (baseline B)."""
    train, test = (f.collect() for f in split_frame(events, split))

    dest_counts = iforest.user_fanout(train)
    train_features = iforest.build_features(train, dest_counts)
    test_features = iforest.align(iforest.build_features(test, dest_counts),
                                  train_features.columns)

    model = iforest.fit_forest(train_features)
    return (test.with_columns(
        pl.Series("score", iforest.score(model, test_features)))
        .sort("score", descending=True))


def score_surprisal(events, split, method="mad", **_opts):
    """Per-user surprisal z-score over the test window."""
    train, test = split_frame(events, split)
    ct = surp.count_tables(train)
    moments = surp.fit_moments(train, ct, method=method)
    scored = surp.standardize(surp.event_surprisal(test, ct), moments)
    return rank_scored(scored.with_columns(score=pl.col("z")))


def _score_windowed(events, split, window_seconds, temperature, observable,
                    method):
    """Per-user z, ranked by a trailing-window observable (see
    surp.window_observable)."""
    train, test = split_frame(events, split)
    ct = surp.count_tables(train)
    moments = surp.fit_moments(train, ct, method=method)
    scored = surp.standardize(surp.event_surprisal(test, ct), moments)
    windowed = surp.window_observable(scored, window_seconds, temperature,
                                      observable)
    return rank_scored(windowed.with_columns(score=pl.col("z_win")))


def score_surprisal_mean(events, split, window_seconds=surp.WINDOW_SECONDS,
                         temperature=None, method="mad", **_opts):
    """Windowed surprisal over the test window.

    Same per-user z as surprisal, ranked by the trailing-window free-energy
    observable of z. temperature=None is the plain window mean; finite T is the
    soft-max free energy (see surp.window_observable).
    """
    return _score_windowed(events, split, window_seconds, temperature, "mean",
                           method)


def score_surprisal_var(events, split, window_seconds=surp.WINDOW_SECONDS,
                        method="mad", **_opts):
    """Windowed variance of surprisal (heat-capacity analog) over the test
    window. Ranks on how much a user's z spread widens in the trailing window.
    """
    return _score_windowed(events, split, window_seconds, None, "var", method)


def score_surprisal_edges(events, split, window_seconds=surp.WINDOW_SECONDS,
                          method="mad", **_opts):
    """Distinct (src_comp, dst_comp, auth_type) count in the trailing window.

    Ranks busyness, not deviation, and reads no z. The sweep's null observable.
    """
    return _score_windowed(events, split, window_seconds, None, "edges", method)


def rank_scored(scored):
    """Rank a scored LazyFrame, keeping only what recall@budget reads."""
    return (scored.select("time", "score", "label")
            .sort(["score", "time"], descending=[True, False])
            .collect())


SCORERS = {
    "novelty": score_novelty,
    "iforest": score_iforest,
    "surprisal": score_surprisal,
    "surprisal-mean": score_surprisal_mean,
    "surprisal-var": score_surprisal_var,
    "surprisal-edges": score_surprisal_edges,
}

# Columns each model reads. The scan is projected to these so the wide auth
# schema never materializes; the surprisal family shares one set.
REQUIRED = {
    "novelty": novelty.NEEDED,
    "iforest": iforest.NEEDED,
    "surprisal": surp.NEEDED,
    "surprisal-mean": surp.NEEDED,
    "surprisal-var": surp.NEEDED,
    "surprisal-edges": surp.NEEDED,
}


def detect(input_path, model, budgets=DEFAULT_BUDGETS, split_fraction=None,
           window_seconds=surp.WINDOW_SECONDS, temperature=None, method="mad"):
    """Score `input_path` with `model`, return (ranked, recall_table, redteam).

    ranked: the model's scored alerts, one row per test event.
    recall_table: recall@budget frame (budget, alerts_used, redteam_caught, recall).
    redteam: test-window red team count, the recall denominator.
    window_seconds: trailing window, windowed surprisal models only.
    temperature: free-energy T for surprisal-mean; None is the window mean.
    method: per-user standardization; the baselines ignore it.
    """
    events = pl.scan_csv(input_path).select(REQUIRED[model])
    if split_fraction is None:
        split = FIRST_REDTEAM_TIME
    else:
        split = split_time_from_fraction(events, split_fraction)

    # every scorer takes the full option set and ignores what it does not read
    ranked = SCORERS[model](events, split, window_seconds=window_seconds,
                            temperature=temperature, method=method)
    redteam = int(eval_window(events, split)
                  .select((pl.col("label") == 1).sum()).collect().item())
    return ranked, recall_at_budget(ranked, redteam, budgets=budgets), redteam


def _run_one(input_path, model, budgets, split_fraction, window_seconds,
             temperature, method):
    """Score one model and print its recall@budget table."""
    ranked, table, redteam = detect(input_path, model, budgets=budgets,
                                    split_fraction=split_fraction,
                                    window_seconds=window_seconds,
                                    temperature=temperature, method=method)
    print(f"model {model}: {ranked.height:,} test events scored; "
          f"{redteam:,} red team")
    print(f"per-day recall@budget (denominator: {redteam:,} red team)")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(table)
    print()


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="labeled auth subset CSV")
    parser.add_argument("--model", required=True, choices=(*SCORERS, "all"),
                        help="detector to score with; 'all' runs each in turn")
    parser.add_argument("--budget", type=int, nargs="+", default=None,
                        help="per-day alert budget(s) to report recall at; "
                             f"default {list(DEFAULT_BUDGETS)}")
    parser.add_argument("--split-fraction", type=float, default=None,
                        help="train/test split at this fraction of the time range "
                             "(0-1); default is the first red team event")
    parser.add_argument("--window", type=int, default=surp.WINDOW_SECONDS,
                        help="windowed-surprisal trailing window in seconds; "
                             f"default {surp.WINDOW_SECONDS}")
    parser.add_argument("--method", choices=("mad", "meanstd", "logz"),
                        default="mad",
                        help="per-user standardization for the surprisal "
                             "family; default mad")
    parser.add_argument("--temperature", type=float, default=None,
                        help="surprisal-mean free-energy T > 0; default (unset) "
                             "is the window mean, small T approaches the max")
    args = parser.parse_args()

    budgets = args.budget
    if budgets is None:
        budgets = list(DEFAULT_BUDGETS)
        print(f"no --budget given, using default {budgets}")

    if args.model == "all":
        # One subprocess per model so peak memory is a single model
        base = [sys.executable, __file__, "--input", args.input,
                "--budget", *map(str, budgets), "--window", str(args.window),
                "--method", args.method]
        if args.split_fraction is not None:
            base += ["--split-fraction", str(args.split_fraction)]
        if args.temperature is not None:
            base += ["--temperature", str(args.temperature)]
        for model in SCORERS:
            subprocess.run(base + ["--model", model], check=True)
    else:
        _run_one(args.input, args.model, budgets, args.split_fraction,
                 args.window, args.temperature, args.method)
