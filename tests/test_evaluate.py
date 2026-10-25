import polars as pl

from src.evaluate import (
    _ranked_by_day,
    recall_at_budget,
    recall_at_budget_global,
)

DAY = 86400


def alerts(rows):
    """rows: (time, score, label) tuples -> scored alerts frame."""
    cols = ["time", "score", "label"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"time": pl.Int64, "score": pl.Float64,
                                "label": pl.Int64})


def test_ranked_by_day_partitions_and_orders():
    ranked = _ranked_by_day(
        alerts([(10, 0.1, 0), (20, 0.9, 0), (DAY + 5, 0.5, 0)]),
        "time", "score", DAY)
    rows = ranked.select("time", "day", "day_rank").rows()
    # day 0 reordered high score first; day 1 restarts the rank at 0.
    assert rows == [(20, 0, 0), (10, 0, 1), (DAY + 5, 1, 0)]


def test_day_rank_breaks_ties_by_time_ascending():
    ranked = _ranked_by_day(
        alerts([(30, 0.5, 0), (10, 0.5, 0), (20, 0.5, 0)]),
        "time", "score", DAY)
    assert ranked.select("time").to_series().to_list() == [10, 20, 30]


def test_recall_at_budget_caps_per_day():
    # day 0: red team event sits at rank 1, caught only at budget >= 2.
    # day 1: lone red team event, always caught.
    scored = alerts([(10, 0.9, 0), (20, 0.8, 1), (30, 0.1, 0), (DAY + 5, 0.5, 1)])
    table = recall_at_budget(scored, total_redteam=2, budgets=[1, 2])
    assert table.filter(pl.col("budget") == 1)["recall"].item() == 0.5
    assert table.filter(pl.col("budget") == 2)["recall"].item() == 1.0


def test_recall_at_budget_alerts_used_capped_by_rows_per_day():
    # day 0 has 2 rows; a budget of 5 cannot use more than the 2 available.
    table = recall_at_budget(alerts([(10, 0.9, 0), (20, 0.8, 0)]),
                             total_redteam=1, budgets=[5])
    assert table["alerts_used"].item() == 2


def test_recall_uses_passed_denominator_not_alerts_label_sum():
    # alerts carry 1 labeled row but the true test window held 4 red team.
    table = recall_at_budget(alerts([(10, 0.9, 1)]), total_redteam=4, budgets=[1])
    assert table["recall"].item() == 0.25


def test_recall_zero_redteam_does_not_divide_by_zero():
    table = recall_at_budget(alerts([(10, 0.9, 0)]), total_redteam=0, budgets=[1])
    assert table["recall"].item() == 0.0


def test_false_positives_and_precision_columns():
    # day 0 top-2 = one red (0.8) + one benign (0.9); 1 FP, precision 1/2.
    scored = alerts([(10, 0.9, 0), (20, 0.8, 1), (30, 0.1, 0), (DAY + 5, 0.5, 1)])
    row = recall_at_budget(scored, total_redteam=2, budgets=[2]).filter(
        pl.col("budget") == 2)
    assert row["alerts_used"].item() == 3       # 2 on day 0, 1 on day 1
    assert row["redteam_caught"].item() == 2
    assert row["false_positives"].item() == 1   # used - caught
    assert abs(row["precision"].item() - 2 / 3) < 1e-12


def test_precision_zero_alerts_does_not_divide_by_zero():
    table = recall_at_budget(alerts([]), total_redteam=1, budgets=[5])
    assert table["precision"].item() == 0.0
    assert table["false_positives"].item() == 0


def test_global_pools_budget_over_test_days():
    # 2 test days, budget 1 -> top 2 pooled: scores 0.9 and 0.8, the 0.8 is red.
    scored = alerts([(10, 0.9, 0), (20, 0.8, 1), (30, 0.1, 0), (DAY + 5, 0.5, 1)])
    table = recall_at_budget_global(scored, total_redteam=2, budgets=[1])
    assert table["total_alerts"].item() == 2
    assert table["recall"].item() == 0.5
