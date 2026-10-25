"""Guards for the per-model NEEDED column projections.

Each detector projects scan_csv to its NEEDED tuple so the wide auth schema
never loads.
"""
import polars as pl

import baseline_iforest
import baseline_novelty
import detect
import make_subsets
import surprisal

DETECTORS = (baseline_novelty, baseline_iforest, surprisal)


def _schema_columns():
    """Canonical auth-subset columns, straight from the writer's CSV header."""
    return set(make_subsets.HEADER.strip().split(","))


def test_needed_is_subset_of_schema():
    schema = _schema_columns()
    for module in DETECTORS:
        missing = set(module.NEEDED) - schema
        assert not missing, f"{module.__name__} NEEDED not in schema: {missing}"


def test_needed_includes_time_and_label():
    # every model splits on time and scores recall against label
    for module in DETECTORS:
        assert "time" in module.NEEDED and "label" in module.NEEDED


def test_detect_required_matches_module_needed():
    assert detect.REQUIRED["novelty"] == baseline_novelty.NEEDED
    assert detect.REQUIRED["iforest"] == baseline_iforest.NEEDED
    assert detect.REQUIRED["surprisal"] == surprisal.NEEDED
    assert detect.REQUIRED["surprisal-mean"] == surprisal.NEEDED


def test_required_covers_every_model():
    assert set(detect.REQUIRED) == set(detect.SCORERS)


def test_surprisal_mean_runs_on_needed_only_frame():
    # a frame carrying only surprisal.NEEDED successfully scores end to end:
    rows = [
        (1, "U1", "C1", "C2", "Kerberos", 0),
        (2, "U1", "C1", "C2", "Kerberos", 0),
        (3, "U1", "C1", "C3", "Kerberos", 0),
        (100, "U1", "C1", "C9", "Kerberos", 1),
        (101, "U1", "C1", "C8", "Kerberos", 1),
    ]
    frame = pl.LazyFrame(
        {c: [r[i] for r in rows] for i, c in enumerate(surprisal.NEEDED)},
        schema={"time": pl.Int64, "src_user": pl.Utf8, "src_comp": pl.Utf8,
                "dst_comp": pl.Utf8, "auth_type": pl.Utf8, "label": pl.Int64})
    ranked = detect.score_surprisal_mean(frame, split=100)
    assert set(ranked.columns) == {"time", "score", "label"}
    assert ranked["score"].is_finite().all()
