import polars as pl

from src.baseline_novelty import (
    PAIR_KEYS,
    assign_tier,
    dest_popularity,
    rank_alerts,
    score_events,
    train_edges,
)

COLUMNS = ["time", "src_user", "src_comp", "dst_comp", "auth_type", "label"]
SCHEMA = {"time": pl.Int64, "src_user": pl.Utf8, "src_comp": pl.Utf8,
          "dst_comp": pl.Utf8, "auth_type": pl.Utf8, "label": pl.Int64}


def auth_lazy_frame(rows):
    """rows: (time, src_user, src_comp, dst_comp, auth_type, label) -> LazyFrame."""
    data = {col: [row[i] for row in rows] for i, col in enumerate(COLUMNS)}
    return pl.LazyFrame(data, schema=SCHEMA)


def edge_frame(edges):
    """edges: (src_comp, dst_comp, auth_type) tuples -> train_edges output."""
    return pl.LazyFrame(
        {"src_comp": [e[0] for e in edges], "dst_comp": [e[1] for e in edges],
         "auth_type": [e[2] for e in edges]},
        schema={"src_comp": pl.Utf8, "dst_comp": pl.Utf8, "auth_type": pl.Utf8},
    )


def pair_frame(pairs):
    """pairs: (dst_comp, auth_type) tuples -> train pair-edge output."""
    return pl.LazyFrame(
        {"dst_comp": [p[0] for p in pairs], "auth_type": [p[1] for p in pairs]},
        schema={"dst_comp": pl.Utf8, "auth_type": pl.Utf8},
    )


def test_train_edges_dedupes_before_split():
    frame = auth_lazy_frame([(1, "U1", "C1", "C2", "Kerberos", 0),
                             (2, "U1", "C1", "C2", "Kerberos", 0),
                             (99, "U1", "C1", "C9", "NTLM", 1)])
    edges = train_edges(frame, split=50).collect()
    assert list(zip(edges["src_comp"], edges["dst_comp"], edges["auth_type"])) \
        == [("C1", "C2", "Kerberos")]


def test_train_edges_custom_keys():
    # pair backoff key: (dst_comp, auth_type) collapses across src_comp
    frame = auth_lazy_frame([(1, "U1", "C1", "C2", "Kerberos", 0),
                             (2, "U1", "C7", "C2", "Kerberos", 0)])
    pairs = train_edges(frame, split=99, keys=PAIR_KEYS).collect()
    assert list(zip(pairs["dst_comp"], pairs["auth_type"])) == [("C2", "Kerberos")]


def test_score_flags_only_unseen_edges():
    frame = auth_lazy_frame([(1, "U1", "C1", "C2", "Kerberos", 0),
                             (2, "U1", "C1", "C9", "Kerberos", 1)])
    alerts = score_events(frame, edge_frame([("C1", "C2", "Kerberos")]),
                          split=0).collect()
    assert alerts["dst_comp"].to_list() == ["C9"]


def test_score_new_auth_type_on_known_pair_is_alert():
    # same src+dst but a new auth_type is an unseen triple
    frame = auth_lazy_frame([(1, "U1", "C1", "C2", "NTLM", 1)])
    alerts = score_events(frame, edge_frame([("C1", "C2", "Kerberos")]),
                          split=0).collect()
    assert alerts["auth_type"].to_list() == ["NTLM"]


def test_score_excludes_train_window():
    frame = auth_lazy_frame([(5, "U1", "C1", "C9", "Kerberos", 0),
                             (15, "U1", "C1", "C9", "Kerberos", 1)])
    alerts = score_events(frame, edge_frame([]), split=10).collect()
    assert alerts["time"].to_list() == [15]


def test_assign_tier_pair_novel_is_zero():
    alerts = auth_lazy_frame([(60, "U1", "C1", "C2", "NTLM", 1)])
    tiered = assign_tier(alerts, pair_frame([("C2", "Kerberos")])).collect()
    assert tiered["tier"].to_list() == [0]


def test_assign_tier_seen_pair_is_one():
    # only src_comp is new; (dst_comp, auth_type) already seen in train
    alerts = auth_lazy_frame([(60, "U1", "C7", "C2", "Kerberos", 1)])
    tiered = assign_tier(alerts, pair_frame([("C2", "Kerberos")])).collect()
    assert tiered["tier"].to_list() == [1]


def test_dest_popularity_counts_distinct_train_users():
    frame = auth_lazy_frame([(1, "U1", "C1", "C2", "Kerberos", 0),
                             (2, "U1", "C1", "C2", "Kerberos", 0),
                             (3, "U2", "C3", "C2", "Kerberos", 0),
                             (99, "U1", "C1", "C9", "NTLM", 1)])
    pop = dest_popularity(frame, split=50).collect().sort("dst_comp")
    assert dict(zip(pop["dst_comp"], pop["train_dest_users"])) == {"C2": 2}


def test_rank_alerts_tier_outranks_rarity():
    # tier 0 on a popular dst still beats tier 1 on a rare dst
    pop = pl.LazyFrame({"dst_comp": ["C2"], "train_dest_users": [9]},
                       schema={"dst_comp": pl.Utf8, "train_dest_users": pl.Int64})
    alerts = pl.LazyFrame(
        {"time": [60, 70], "dst_comp": ["C2", "C9"], "tier": [0, 1]},
        schema={"time": pl.Int64, "dst_comp": pl.Utf8, "tier": pl.Int64},
    )
    ranked = rank_alerts(alerts, pop).collect()
    assert ranked["dst_comp"].to_list() == ["C2", "C9"]
    assert ranked["tier"].to_list() == [0, 1]


def test_rank_alerts_rare_dest_first_within_tier():
    pop = pl.LazyFrame({"dst_comp": ["C2"], "train_dest_users": [2]},
                       schema={"dst_comp": pl.Utf8, "train_dest_users": pl.Int64})
    alerts = pl.LazyFrame(
        {"time": [60, 70], "dst_comp": ["C2", "C9"], "tier": [0, 0]},
        schema={"time": pl.Int64, "dst_comp": pl.Utf8, "tier": pl.Int64},
    )
    ranked = rank_alerts(alerts, pop).collect()
    assert ranked["dst_comp"].to_list() == ["C9", "C2"]
    assert ranked["score"].to_list() == [1.0, 1 / 3]


def test_rank_alerts_ties_broken_by_time():
    pop = pl.LazyFrame({"dst_comp": ["C1"], "train_dest_users": [1]},
                       schema={"dst_comp": pl.Utf8, "train_dest_users": pl.Int64})
    alerts = pl.LazyFrame(
        {"time": [70, 60], "dst_comp": ["C9", "C8"], "tier": [0, 0]},
        schema={"time": pl.Int64, "dst_comp": pl.Utf8, "tier": pl.Int64},
    )
    ranked = rank_alerts(alerts, pop).collect()
    assert ranked["time"].to_list() == [60, 70]
