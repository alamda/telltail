#!/usr/bin/env python3
"""Baseline B: Isolation Forest over auth/logon features.

Fit forest on train window, score test window.
Anomaly score is -score_samples: higher -> more isolated/sus

Features: one-hot auth_type, logon_type, auth_orientation, outcome,
plus hour of day and the user's train fan-out (distinct destinations).

Schema (from make_subsets.py):
  time, src_user, dst_user, src_comp, dst_comp, auth_type,
  logon_type, auth_orientation, outcome, label
"""
import polars as pl
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["auth_type", "logon_type", "auth_orientation", "outcome"]
USER_COL = "src_user"
DEST_COL = "dst_comp"

# Columns features and output use. Project scan_csv to these; drops src_comp,
# dst_user so the wide auth schema never loads.
NEEDED = ("time", USER_COL, DEST_COL, *FEATURE_COLS, "label")


def encode(frame, cols=FEATURE_COLS):
    """One-hot the categorical columns into a numeric feature frame."""
    return frame.select(cols).to_dummies(cols)


def user_fanout(train, user_col=USER_COL, dest_col=DEST_COL):
    """Distinct destinations per user in train.

    Computed from train, applied to test."""
    return (train.group_by(user_col)
            .agg(pl.col(dest_col).n_unique().alias("user_dest_count")))


def build_features(frame, dest_counts, user_col=USER_COL):
    """One-hot categoricals plus hour and user_dest_count.

    hour:            (time // 3600) % 24.
    user_dest_count: train fan-out for the user, 0 if not seen in train.

    Select only needed columns before joining.
    maintain_order keeps the joined rows in `frame` order
    """
    joined = (frame.select("time", user_col, *FEATURE_COLS)
              .join(dest_counts, on=user_col, how="left", maintain_order="left"))
    behavioral = joined.select(
        ((pl.col("time") // 3600) % 24).cast(pl.UInt8).alias("hour"),
        pl.col("user_dest_count").fill_null(0).cast(pl.UInt32),
    )
    return pl.concat([encode(joined), behavioral], how="horizontal")


def align(features, columns):
    """Match feature frame to fixed column set. Add missing as 0, drop extras."""
    present = set(features.columns)
    return features.select(
        [pl.col(c) if c in present else pl.lit(0, dtype=pl.UInt8).alias(c)
         for c in columns])


def matrix(features):
    """Dense float32 array for sklearn."""
    return features.select(pl.all().cast(pl.Float32)).to_numpy()


def fit_forest(train_features, seed=0):
    """Fit Isolation Forest on encoded train rows."""
    model = IsolationForest(random_state=seed)
    model.fit(matrix(train_features))
    return model


def score(model, test_features):
    """Anomaly score per row. Higher is more isolated."""
    return -model.score_samples(matrix(test_features))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from evaluate import recall_at_budget
    from timesplit import (FIRST_REDTEAM_TIME, split_time_from_fraction,
                           split_frame)

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

    train, test = (f.collect() for f in split_frame(events, split))

    # Align test columns to train encoding
    dest_counts = user_fanout(train)
    train_features = build_features(train, dest_counts)
    test_features = align(build_features(test, dest_counts),
                          train_features.columns)

    model = fit_forest(train_features)
    scored = test.with_columns(pl.Series("score", score(model, test_features))) \
                 .sort("score", descending=True)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "iforest_alerts.csv"
    scored.write_csv(out_path)

    redteam = int((test["label"] == 1).sum())
    print(f"test {test.height} rows; red team {redteam}")
    print(f"scored alerts written to {out_path}")

    print(f"\nper-day recall@budget (denominator: {redteam} red team)")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(recall_at_budget(scored, redteam))

    print("\ntop scored:")
    top = scored.head(10).select("time", "src_user", "dst_comp",
                                 "auth_type", "score", "label")
    with pl.Config(tbl_rows=10, tbl_hide_dataframe_shape=True):
        print(top)
