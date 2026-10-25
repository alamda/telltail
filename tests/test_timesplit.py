import polars as pl
import pytest

from src.timesplit import (FIRST_REDTEAM_TIME, split_time_from_fraction,
                           split_frame, train_window, eval_window)


def time_frame(times):
    return pl.LazyFrame({"time": times}, schema={"time": pl.Int64})


def times(frame):
    return frame.collect()["time"].to_list()


def test_split_time_from_fraction_midpoint():
    assert split_time_from_fraction(time_frame([0, 100]), 0.5) == 50


@pytest.mark.parametrize("verboten", [1.5, -0.1])
def test_split_time_from_fraction_rejects_out_of_range(verboten):
    with pytest.raises(ValueError):
        split_time_from_fraction(time_frame([0, 100]), verboten)


def test_split_frame_boundary():
    train, test = split_frame(time_frame([99, 100, 100, 101]), 100)
    assert times(train) == [99]
    assert times(test) == [100, 100, 101]


def test_split_frame_matches_windows():
    frame = time_frame([99, 100, 101])
    train, test = split_frame(frame, 100)
    assert times(train) == times(train_window(frame, 100))
    assert times(test) == times(eval_window(frame, 100))


def test_first_redteam_event_excluded_from_train():
    frame = pl.LazyFrame({"time": [FIRST_REDTEAM_TIME - 1, FIRST_REDTEAM_TIME],
                          "label": [0, 1]})
    train, test = split_frame(frame, FIRST_REDTEAM_TIME)
    assert train.collect()["label"].sum() == 0
    assert test.collect()["label"].to_list() == [1]
