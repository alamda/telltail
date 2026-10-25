import polars as pl

from src.detect import score_iforest

SPLIT = 100


def events():
    """20 train rows before SPLIT, 3 test rows from SPLIT on."""
    times = list(range(80, 100)) + [100, 101, 102]
    n = len(times)
    return pl.LazyFrame({
        "time": times,
        "src_user": ["U1"] * n,
        "dst_comp": ["C1", "C2"] * 10 + ["C1", "C2", "C3"],
        "auth_type": ["Kerberos"] * n,
        "logon_type": ["Network"] * n,
        "auth_orientation": ["LogOn"] * n,
        "outcome": ["Success"] * n,
        "label": [0] * (n - 1) + [1],
    })


def test_score_iforest_returns_test_rows_only():
    ranked = score_iforest(events(), SPLIT)
    # Check for eager DataFrame
    assert isinstance(ranked, pl.DataFrame)
    # Only test rows returned
    assert sorted(ranked["time"].to_list()) == [100, 101, 102]
    # Red-team row survives scoring
    assert ranked["label"].sum() == 1


def test_score_iforest_sorted_by_score():
    ranked = score_iforest(events(), SPLIT)
    assert ranked["score"].is_sorted(descending=True)
