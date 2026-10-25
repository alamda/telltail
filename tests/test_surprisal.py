import polars as pl

from src.surprisal import (
    count_tables,
    event_surprisal,
    fit_moments,
    standardize,
    window_observable,
)

COLUMNS = ["time", "src_user", "src_comp", "dst_comp", "auth_type", "label"]
SCHEMA = {"time": pl.Int64, "src_user": pl.Utf8, "src_comp": pl.Utf8,
          "dst_comp": pl.Utf8, "auth_type": pl.Utf8, "label": pl.Int64}


def auth_lazy_frame(rows):
    """rows: (time, src_user, src_comp, dst_comp, auth_type, label) -> LazyFrame."""
    data = {col: [row[i] for row in rows] for i, col in enumerate(COLUMNS)}
    return pl.LazyFrame(data, schema=SCHEMA)


SAMPLE = auth_lazy_frame([
    (1, "U1", "C1", "C2", "Kerberos", 0),
    (2, "U1", "C1", "C2", "Kerberos", 0),   # repeat edge: n=2 for U1 (C1,C2)
    (3, "U1", "C1", "C3", "NTLM", 0),
    (4, "U2", "C5", "C2", "Kerberos", 0),   # different user, shared dst C2
    (5, "U2", "C5", "C9", "Kerberos", 0),
])


def test_counts_conserve_total():
    # every axis counts the same rows, so all totals equal N
    ct = count_tables(SAMPLE)
    assert ct.N == 5
    assert ct.pop_dst["n"].sum() == ct.N
    assert ct.pop_edge["n"].sum() == ct.N
    assert ct.pop_auth["n"].sum() == ct.N
    assert ct.user_totals["n_u"].sum() == ct.N


def test_vocab_sizes():
    ct = count_tables(SAMPLE)
    assert ct.D == 3    # dst_comp: C2, C3, C9
    assert ct.E == 4    # edges: (C1,C2),(C1,C3),(C5,C2),(C5,C9)
    assert ct.A == 2    # auth_type: Kerberos, NTLM


def test_user_edge_repeat_counted():
    ct = count_tables(SAMPLE)
    row = ct.user_edge.filter(
        (pl.col("src_user") == "U1")
        & (pl.col("src_comp") == "C1")
        & (pl.col("dst_comp") == "C2"))
    assert row["n"].item() == 2


def test_user_totals_match_per_axis_sums():
    # N_u must equal the sum of any per-user axis for that user
    ct = count_tables(SAMPLE)
    per_user_dst = ct.user_dst.group_by("src_user").agg(pl.col("n").sum().alias("s"))
    joined = ct.user_totals.join(per_user_dst, on="src_user")
    assert (joined["n_u"] == joined["s"]).all()


def test_population_pools_across_users():
    # dst C2 seen by U1 (x2) and U2 (x1) -> pop count 3, split over two user rows
    ct = count_tables(SAMPLE)
    assert ct.pop_dst.filter(pl.col("dst_comp") == "C2")["n"].item() == 3
    assert ct.user_dst.filter(pl.col("dst_comp") == "C2").height == 2


# --- pass 2: surprisal, moments, standardization ---

# U1: 3x (C1,C2), 1x (C1,C3). U2: one each to C7, C8. All Kerberos.
TRAIN = auth_lazy_frame([
    (1, "U1", "C1", "C2", "Kerberos", 0),
    (2, "U1", "C1", "C2", "Kerberos", 0),
    (3, "U1", "C1", "C2", "Kerberos", 0),
    (4, "U1", "C1", "C3", "Kerberos", 0),
    (5, "U2", "C5", "C7", "Kerberos", 0),
    (6, "U2", "C5", "C8", "Kerberos", 0),
])


def test_surprisal_is_finite_and_positive():
    ct = count_tables(TRAIN)
    out = event_surprisal(TRAIN, ct).collect()
    assert out["s"].is_finite().all()
    assert (out["s"] > 0).all()   # every prob < 1, so -log > 0


def test_novel_edge_scores_above_known_edge():
    # known user U1: novel dst C99 scores higher than the seen dst C2
    ct = count_tables(TRAIN)
    test = auth_lazy_frame([(10, "U1", "C1", "C2", "Kerberos", 0),
                            (11, "U1", "C1", "C99", "Kerberos", 1)])
    out = event_surprisal(test, ct).collect()
    by_dst = dict(zip(out["dst_comp"], out["s"]))
    assert by_dst["C99"] > by_dst["C2"]


def test_unseen_user_backs_off_to_population():
    # no train history: N_u=0, backoff collapses to population, no NaN
    ct = count_tables(TRAIN)
    test = auth_lazy_frame([(10, "U9", "C1", "C2", "Kerberos", 0)])
    out = event_surprisal(test, ct).collect()
    assert out["s"].is_finite().all()


def test_moments_have_pooled_fallback():
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct)
    assert set(m.per_user["src_user"]) == {"U1", "U2"}
    assert m.scale is not None and m.scale >= 0


def test_zero_scale_user_rescued_by_pooled():
    # U1's train surprisals are mostly tied -> per-user MAD is 0; z must stay
    # finite via the pooled-scale fallback rather than dividing by eps
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct, method="mad")
    assert m.per_user.filter(pl.col("src_user") == "U1")["scale"].item() == 0.0
    test = auth_lazy_frame([(10, "U1", "C1", "C99", "Kerberos", 1)])
    z = standardize(event_surprisal(test, ct), m).collect()
    assert z["z"].is_finite().all()
    assert z["scale"].item() == m.scale   # rescued to pooled, not 0


def test_logz_moments_finite_and_positive_scale_for_tied_user():
    # U1's train surprisals tie (MAD=0), but the log-Z temperature is derived
    # from the fitted distribution, so every user gets a finite, nonzero scale
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct, method="logz")
    assert set(m.per_user["src_user"]) == {"U1", "U2"}
    assert m.per_user["center"].is_finite().all()
    assert m.per_user["scale"].is_finite().all()
    assert (m.per_user["scale"] > 0).all()   # no zero-scale floor needed
    u1 = m.per_user.filter(pl.col("src_user") == "U1")["scale"].item()
    assert u1 > 0.0


def test_logz_unseen_user_gets_population_moments():
    # the analytic formula collapses to the pooled center/scale at N_u=0, so an
    # absent user standardizes against the population temperature with no NaN
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct, method="logz")
    test = auth_lazy_frame([(10, "U9", "C1", "C2", "Kerberos", 0)])
    z = standardize(event_surprisal(test, ct), m).collect()
    assert z["center"].item() == m.center
    assert z["scale"].item() == m.scale
    assert z["z"].is_finite().all()


def test_standardize_unseen_user_uses_pooled_center_scale():
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct)
    test = auth_lazy_frame([(10, "U9", "C1", "C2", "Kerberos", 0)])
    z = standardize(event_surprisal(test, ct), m).collect()
    assert z["center"].item() == m.center
    assert z["scale"].item() == m.scale


# --- windowed mean observable (surprisal density) ---

def _scored_frame(rows):
    """count -> standardize a test frame, returning the z-carrying LazyFrame."""
    ct = count_tables(TRAIN)
    m = fit_moments(TRAIN, ct)
    return standardize(event_surprisal(auth_lazy_frame(rows), ct), m)


def test_window_is_mean_of_z_within_window_per_user():
    # z_win at each event is the mean z over that user's trailing window
    scored = _scored_frame([
        (100, "U1", "C1", "C99", "Kerberos", 1),
        (200, "U1", "C1", "C98", "Kerberos", 1),
        (300, "U1", "C1", "C97", "Kerberos", 1),
        (150, "U2", "C5", "C7", "Kerberos", 0),
    ])
    out = window_observable(scored, window_seconds=3600).collect()
    by = {(u, t): (z, w) for u, t, z, w
          in zip(out["src_user"], out["time"], out["z"], out["z_win"])}
    # last U1 event: mean of all three z in the window
    z_mean = sum(by[("U1", t)][0] for t in (100, 200, 300)) / 3
    assert abs(by[("U1", 300)][1] - z_mean) < 1e-9
    # U2's window holds only itself, so its mean is its own z
    assert by[("U2", 150)][1] == by[("U2", 150)][0]


def test_window_excludes_events_past_window():
    # gap > window: the later event's mean is just its own z
    scored = _scored_frame([
        (100, "U1", "C1", "C99", "Kerberos", 1),
        (100 + 3601, "U1", "C1", "C98", "Kerberos", 1),
    ])
    out = window_observable(scored, window_seconds=3600).collect()
    by = dict(zip(out["time"], zip(out["z_win"], out["z"])))
    late = by[100 + 3601]
    assert late[0] == late[1]


def test_window_mean_ignores_volume():
    # sustained-anomaly user (few, all high-z) beats a high-volume mostly-normal
    # user: the mean stays up for the former, averages down for the latter
    scored = _scored_frame([
        (10, "U1", "C1", "C99", "Kerberos", 1),   # both novel -> high z
        (20, "U1", "C1", "C98", "Kerberos", 1),
    ] + [(1000 + i, "U2", "C5", "C7", "Kerberos", 0) for i in range(20)])  # seen -> low z
    out = window_observable(scored, window_seconds=3600).collect()
    anomalous = out.filter(pl.col("src_user") == "U1")["z_win"].max()
    high_volume = out.filter(pl.col("src_user") == "U2")["z_win"].max()
    assert anomalous > high_volume


# --- free-energy window observable ---

FREE_ENERGY_ROWS = [
    (100, "U1", "C1", "C99", "Kerberos", 1),
    (200, "U1", "C1", "C98", "Kerberos", 1),
    (300, "U1", "C1", "C2", "Kerberos", 0),
]


def test_large_temperature_recovers_mean():
    # T -> inf: G_T collapses to the window mean (the default observable)
    scored = _scored_frame(FREE_ENERGY_ROWS)
    mean = window_observable(scored, window_seconds=3600).collect()
    hot = window_observable(scored, window_seconds=3600, temperature=1e6).collect()
    m = dict(zip(mean["time"], mean["z_win"]))
    h = dict(zip(hot["time"], hot["z_win"]))
    for t in m:
        assert abs(m[t] - h[t]) < 1e-3


def test_large_temperature_recovers_mean_across_users():
    # The overflow shift is a per-user max. If it is not removed in the same
    # window context, each user's G_T keeps a constant offset, which is
    # invisible with one user but reorders the ranking across users.
    rows = FREE_ENERGY_ROWS + [
        (100, "U2", "C3", "C97", "Kerberos", 1),
        (200, "U2", "C3", "C96", "Kerberos", 1),
        (300, "U2", "C3", "C4", "Kerberos", 0),
    ]
    scored = _scored_frame(rows)
    mean = window_observable(scored, window_seconds=3600).collect()
    hot = window_observable(scored, window_seconds=3600, temperature=1e6).collect()
    key = ["src_user", "time"]
    joined = mean.select(*key, "z_win").join(
        hot.select(*key, pl.col("z_win").alias("hot")), on=key)
    assert joined.height == len(rows)
    for row in joined.iter_rows(named=True):
        assert abs(row["z_win"] - row["hot"]) < 1e-3


def test_small_temperature_approaches_window_max():
    # T -> 0: G_T approaches the running max of z over the window
    scored = _scored_frame(FREE_ENERGY_ROWS)
    cold = window_observable(scored, window_seconds=3600, temperature=1e-4).collect()
    running = cold.sort("time")
    z, w = running["z"].to_list(), running["z_win"].to_list()
    for i in range(len(z)):
        assert abs(w[i] - max(z[: i + 1])) < 1e-3


def test_free_energy_between_mean_and_max():
    # a finite T sits between the window mean and max at the last event
    scored = _scored_frame(FREE_ENERGY_ROWS)
    last = lambda df: df.sort("time")["z_win"].to_list()[-1]
    z = _scored_frame(FREE_ENERGY_ROWS).collect()["z"].to_list()
    g = last(window_observable(scored, window_seconds=3600, temperature=1.0).collect())
    assert sum(z) / len(z) <= g <= max(z) + 1e-9


def test_nonpositive_temperature_rejected():
    scored = _scored_frame(FREE_ENERGY_ROWS)
    import pytest
    with pytest.raises(ValueError):
        window_observable(scored, temperature=0.0).collect()


# --- variance window observable (heat-capacity analog) ---


def test_window_var_is_variance_of_z_within_window():
    scored = _scored_frame([
        (100, "U1", "C1", "C99", "Kerberos", 1),
        (200, "U1", "C1", "C98", "Kerberos", 1),
        (300, "U1", "C1", "C2", "Kerberos", 0),
    ])
    out = window_observable(scored, window_seconds=3600,
                            observable="var").collect().sort("time")
    z = out["z"].to_list()
    # last event's z_win is the sample variance over the whole window
    mean = sum(z) / len(z)
    var = sum((x - mean) ** 2 for x in z) / (len(z) - 1)
    assert abs(out["z_win"].to_list()[-1] - var) < 1e-9


def test_window_var_single_event_is_zero():
    # a lone event in its window has no spread
    scored = _scored_frame([(100, "U2", "C5", "C7", "Kerberos", 0)])
    out = window_observable(scored, window_seconds=3600,
                            observable="var").collect()
    assert out["z_win"].to_list() == [0.0]


def test_window_edges_counts_distinct_edge_auth():
    # z_win is the distinct (edge, auth_type) count over the trailing window;
    # a repeated edge does not add, a new auth_type on the same edge does
    scored = _scored_frame([
        (100, "U1", "C1", "C99", "Kerberos", 1),
        (200, "U1", "C1", "C99", "Kerberos", 1),  # dup edge+auth, no new count
        (300, "U1", "C1", "C98", "Kerberos", 1),  # new edge
        (400, "U1", "C1", "C98", "NTLM", 1),      # same edge, new auth
    ])
    out = window_observable(scored, window_seconds=3600,
                            observable="edges").collect().sort("time")
    assert out["z_win"].to_list() == [1, 1, 2, 3]


def test_window_edges_excludes_events_past_window():
    # an edge older than the window does not count toward the current one
    scored = _scored_frame([
        (100, "U1", "C1", "C99", "Kerberos", 1),
        (100 + 4000, "U1", "C1", "C98", "Kerberos", 1),  # >3600s later
    ])
    out = window_observable(scored, window_seconds=3600,
                            observable="edges").collect().sort("time")
    assert out["z_win"].to_list() == [1, 1]


def test_unknown_observable_rejected():
    scored = _scored_frame(FREE_ENERGY_ROWS)
    import pytest
    with pytest.raises(ValueError):
        window_observable(scored, observable="entropy").collect()
