"""Shared train/test time split for the baselines and the surprisal model."""
import polars as pl

# First red team event. Train window: everything before first red team event.
FIRST_REDTEAM_TIME = 150885


def split_time_from_fraction(frame, fraction):
    """Timestamp at `fraction` of [min_time, max_time] (0.5 = midpoint)."""
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    bounds = frame.select(pl.col("time").min().alias("low"),
                          pl.col("time").max().alias("high")).collect()
    low, high = bounds["low"][0], bounds["high"][0]
    return low + fraction * (high - low)


def split_frame(frame, split):
    """(train, test) split at split."""
    return (train_window(frame, split), eval_window(frame, split))


def train_window(frame, split):
    """Rows with time < split."""
    return frame.filter(pl.col("time") < split)


def eval_window(frame, split):
    """Rows with time >= split."""
    return frame.filter(pl.col("time") >= split)
