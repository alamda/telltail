import polars as pl

from src.baseline_iforest import (
    align,
    build_features,
    encode,
    fit_forest,
    score,
    user_fanout,
)

FEATURES = pl.DataFrame({
    "auth_type": ["Kerberos", "Kerberos", "NTLM"],
    "logon_type": ["Network", "Network", "Interactive"],
    "auth_orientation": ["LogOn", "LogOn", "LogOn"],
    "outcome": ["Success", "Success", "Success"],
})


def test_encode_one_hot_columns():
    encoded = encode(FEATURES)
    assert "auth_type_Kerberos" in encoded.columns
    assert "auth_type_NTLM" in encoded.columns
    # confirm one-hot rows sum to the number of source columns
    assert encoded.sum_horizontal().to_list() == [4, 4, 4]


def test_align_adds_missing_and_drops_extra():
    features = pl.DataFrame({"a": [1], "b": [1]})
    aligned = align(features, ["a", "c"])
    assert aligned.columns == ["a", "c"]
    assert aligned.row(0) == (1, 0)


def test_score_higher_for_isolated_row():
    train = pl.DataFrame({
        "auth_type": ["Kerberos"] * 18 + ["NTLM"] * 2,
        "logon_type": ["Network"] * 18 + ["Interactive"] * 2,
        "auth_orientation": ["LogOn"] * 20,
        "outcome": ["Success"] * 20,
    })
    test = pl.DataFrame({
        "auth_type": ["Kerberos", "NTLM"],
        "logon_type": ["Network", "Interactive"],
        "auth_orientation": ["LogOn", "LogOff"],
        "outcome": ["Success", "Fail"],
    })
    train_features = encode(train)
    model = fit_forest(train_features)
    scores = score(model, align(encode(test), train_features.columns))
    assert scores[1] > scores[0]


def auth_frame(rows):
    """rows: (time, src_user, dst_comp, auth_type) -> full-schema DataFrame."""
    return pl.DataFrame({
        "time": [r[0] for r in rows],
        "src_user": [r[1] for r in rows],
        "dst_comp": [r[2] for r in rows],
        "auth_type": [r[3] for r in rows],
        "logon_type": ["Network"] * len(rows),
        "auth_orientation": ["LogOn"] * len(rows),
        "outcome": ["Success"] * len(rows),
    })


def test_user_fanout_counts_distinct_dests():
    train = auth_frame([(1, "U1", "C2", "Kerberos"), (2, "U1", "C3", "Kerberos"),
                        (3, "U2", "C2", "Kerberos")])
    dest_counts = user_fanout(train)
    counts = dict(zip(dest_counts["src_user"], dest_counts["user_dest_count"]))
    assert counts == {"U1": 2, "U2": 1}


def test_build_features_hour_from_time():
    dest_counts = user_fanout(auth_frame([(0, "U1", "C2", "Kerberos")]))
    # 3600 s = hour 1, 90000 s = 25 h % 24 = 1
    feats = build_features(auth_frame([(3600, "U1", "C2", "Kerberos"),
                                       (90000, "U1", "C2", "Kerberos")]),
                           dest_counts)
    assert feats["hour"].to_list() == [1, 1]


def test_build_features_fanout_and_cold_start():
    train = auth_frame([(1, "U1", "C2", "Kerberos"), (2, "U1", "C3", "Kerberos")])
    dest_counts = user_fanout(train)
    test = auth_frame([(10, "U1", "C2", "Kerberos"),    # known user
                       (12, "U7", "C2", "Kerberos")])   # new user
    feats = build_features(test, dest_counts)
    assert feats["user_dest_count"].to_list() == [2, 0]
