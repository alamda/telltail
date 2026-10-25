#!/usr/bin/env python3
"""Per-user maximum-entropy surprisal model.

Score each auth by its negative log-probability under a per-user model of normal
behavior, backed off toward the population when a user's history is sparse.

Training is two passes over the train window:
  pass 1: tally histograms per axis.
  pass 2: per-user surprisal moments for z-standardization.

Schema (from make_subsets.py):
  time, src_user, dst_user, src_comp, dst_comp, auth_type,
  logon_type, auth_orientation, outcome, label
"""
from dataclasses import dataclass

import polars as pl

USER = "src_user"
EDGE_KEYS = ("src_comp", "dst_comp")
WINDOW_SECONDS = 3600  # trailing window for the windowed-mean observable, 1 hour

# Columns this model reads. Project scan_csv to these so the wide auth schema
# (dst_user, logon_type, auth_orientation, outcome) never loads.
NEEDED = ("time", USER, *EDGE_KEYS, "auth_type", "label")


@dataclass
class CountTables:
    """Train-window histograms for the surprisal model.

    Population tables pool all users. User tables add the USER key. Counts stay
    raw so the smoothing/backoff params retune without refitting.
    """
    pop_edge: pl.DataFrame     # src_comp, dst_comp   -> n
    pop_dst: pl.DataFrame      # dst_comp             -> n
    pop_auth: pl.DataFrame     # auth_type            -> n
    user_edge: pl.DataFrame    # src_user, src_comp, dst_comp -> n
    user_dst: pl.DataFrame     # src_user, dst_comp   -> n
    user_auth: pl.DataFrame    # src_user, auth_type  -> n
    user_totals: pl.DataFrame  # src_user             -> n_u
    N: int                     # total train events
    D: int                     # distinct destinations (dst vocab size)
    E: int                     # distinct edges (edge vocab size)
    A: int                     # distinct auth_types (auth vocab size)


@dataclass
class Hyperparams:
    """Smoothing (alpha) and backoff (kappa) params, tuned on train.

    Edges are sparser than bare dst -> kappa_edge > kappa_dst.
    """
    alpha_edge: float = 0.5
    kappa_edge: float = 20.0
    alpha_auth: float = 0.5
    kappa_auth: float = 5.0
    alpha_dst: float = 0.5
    kappa_dst: float = 10.0


def _counts(train, keys):
    """Count rows per key combination."""
    return train.group_by(list(keys)).agg(pl.len().alias("n"))


def _backoff_prob(user_count, pop_count, user_total, N, vocab, alpha, kappa):
    """Jelinek-Mercer backoff expr: user frequency shrunk toward the population.

    lambda = N_u/(N_u+kappa) trusts the user's history once it outweighs kappa
    pseudo-counts. Users with N_u->0 collapse to the add-alpha population estimate.
    """
    N_u = pl.col(user_total)
    lam = N_u / (N_u + kappa)
    p_pop = (pl.col(pop_count) + alpha) / (N + alpha * vocab)
    # guard N_u=0: lambda is 0 there, but 0/0 would poison the blend with NaN
    user_frac = pl.when(N_u > 0).then(pl.col(user_count) / N_u).otherwise(0.0)
    return lam * user_frac + (1 - lam) * p_pop


def count_tables(train):
    """Build all train-window count tables."""
    # Train data only, not full stream
    train = train.lazy()

    pop_edge = _counts(train, EDGE_KEYS)
    pop_dst = _counts(train, ("dst_comp",))
    pop_auth = _counts(train, ("auth_type",))
    user_edge = _counts(train, (USER, *EDGE_KEYS))
    user_dst = _counts(train, (USER, "dst_comp"))
    user_auth = _counts(train, (USER, "auth_type"))
    user_totals = train.group_by(USER).agg(pl.len().alias("n_u"))

    (pop_edge, pop_dst, pop_auth, user_edge, user_dst, user_auth,
     user_totals) = pl.collect_all([pop_edge, pop_dst, pop_auth, user_edge,
                                    user_dst, user_auth, user_totals])

    return CountTables(
        pop_edge=pop_edge, pop_dst=pop_dst, pop_auth=pop_auth,
        user_edge=user_edge, user_dst=user_dst, user_auth=user_auth,
        user_totals=user_totals,
        N=pop_dst["n"].sum(),
        D=pop_dst.height,
        E=pop_edge.height,
        A=pop_auth.height,
    )


def event_surprisal(events, ct, hyper=Hyperparams()):
    """Attach to frame per-event surprisal s = -log p_u(r,d) - log p_u(a).

    Left-joins the count tables onto events, backs each axis off toward the
    population, and adds the p_u_edge, p_u_auth, and s columns. Runs on train
    (to fit moments) and on test (to score). Pass the same fitted `ct` to both.
    """
    lf = (events.lazy()
          .join(ct.user_edge.lazy().rename({"n": "nu_e"}),
                on=[USER, *EDGE_KEYS], how="left")
          .join(ct.pop_edge.lazy().rename({"n": "n_e"}),
                on=list(EDGE_KEYS), how="left")
          .join(ct.user_auth.lazy().rename({"n": "nu_a"}),
                on=[USER, "auth_type"], how="left")
          .join(ct.pop_auth.lazy().rename({"n": "n_a"}),
                on="auth_type", how="left")
          .join(ct.user_totals.lazy(), on=USER, how="left")
          .with_columns(pl.col("nu_e", "n_e", "nu_a", "n_a", "n_u").fill_null(0)))

    p_u_e = _backoff_prob("nu_e", "n_e", "n_u", ct.N, ct.E,
                          hyper.alpha_edge, hyper.kappa_edge)
    p_u_a = _backoff_prob("nu_a", "n_a", "n_u", ct.N, ct.A,
                          hyper.alpha_auth, hyper.kappa_auth)
    return lf.with_columns(p_u_edge=p_u_e, p_u_auth=p_u_a).with_columns(
        s=-pl.col("p_u_edge").log() - pl.col("p_u_auth").log())


@dataclass
class Moments:
    """Per-user surprisal center/scale, with pooled-train fallback for users
    absent from train."""
    per_user: pl.DataFrame   # src_user -> center, scale
    center: float
    scale: float


def _center_scale(method):
    """(center, scale) aggregation exprs over the surprisal column `s`.

    'mad' (default): median and 1.4826*MAD, robust to the right-skewed tail.
    'meanstd': mean and std, pulled by outliers but the textbook z-score.
    """
    if method == "mad":
        center = pl.col("s").median()
        scale = 1.4826 * (pl.col("s") - pl.col("s").median()).abs().median()
    elif method == "meanstd":
        center = pl.col("s").mean()
        scale = pl.col("s").std()
    else:
        raise ValueError(f"unknown method {method!r}")
    return center.alias("center"), scale.alias("scale")


def _pop_axis_moments(pop_tbl, n_col, N, vocab, alpha):
    """One axis's population surprisal moments from log Z.

    Population probability p_pop(x) = (n_x + alpha)/(N + alpha*vocab), supported
    on the `vocab` values seen in train (the add-alpha mass sums to 1 over them).
    surprisal_pop(x) = -log p_pop(x) is the energy of value x under the
    population model.

    Returns the population table with p_pop and surprisal_pop columns attached,
    plus two scalars: the mean and the mean-square of surprisal under p_pop,
        pop_mean_surprisal   = sum_x p_pop(x) * surprisal_pop(x)
        pop_meansq_surprisal = sum_x p_pop(x) * surprisal_pop(x)^2
    """
    denom = N + alpha * vocab
    pop = pop_tbl.with_columns(
        p_pop=(pl.col(n_col) + alpha) / denom
    ).with_columns(surprisal_pop=-pl.col("p_pop").log())
    moments = pop.select(
        pop_mean_surprisal=(pl.col("p_pop") * pl.col("surprisal_pop")).sum(),
        pop_meansq_surprisal=(pl.col("p_pop") * pl.col("surprisal_pop") ** 2).sum())
    return (pop, moments["pop_mean_surprisal"][0],
            moments["pop_meansq_surprisal"][0])


def _user_axis_moments(user_tbl, pop, keys, nu_col, user_totals, kappa,
                       pop_mean_surprisal, pop_meansq_surprisal):
    """Per-user mean and mean-square surprisal on one axis, from log Z.

    Under the user's backed-off model p_user(x) = lam*(n_ux/N_u) + w_pop*p_pop(x),
    with backoff weight lam = N_u/(N_u+kappa) and population weight w_pop = 1-lam.
    The user's surprisal moments split into two parts:

      seen part:   summed directly over the k_u values the user has in train,
                   using p_user and its surprisal there.
      unseen part: every value the user never sees has p_user(x) = w_pop*p_pop(x),
                   so surprisal(x) = -log(w_pop) + surprisal_pop(x). That whole
                   block folds into closed form from the population scalars, so
                   we never enumerate the vocab. Cost is O(distinct user-values).

    Returns a frame src_user -> mean_surprisal, meansq_surprisal.
    """
    lam = pl.col("n_u") / (pl.col("n_u") + kappa)
    w_pop = 1 - lam
    p_user = lam * (pl.col(nu_col) / pl.col("n_u")) + w_pop * pl.col("p_pop")
    per_user = (
        user_tbl.lazy()
        .join(pop.select(*keys, "p_pop", "surprisal_pop").lazy(),
              on=list(keys), how="left")
        .join(user_totals.lazy(), on=USER, how="left")
        .with_columns(p_user=p_user, w_pop=w_pop)
        .with_columns(surprisal_user=-pl.col("p_user").log())
        .group_by(USER)
        .agg(
            # seen-value contributions to the user's surprisal moments
            seen_mean=(pl.col("p_user") * pl.col("surprisal_user")).sum(),
            seen_meansq=(pl.col("p_user") * pl.col("surprisal_user") ** 2).sum(),
            # population statistics over just the user's seen values, subtracted
            # from the population-wide scalars to isolate the unseen block below
            pop_mass_seen=pl.col("p_pop").sum(),
            pop_mean_seen=(pl.col("p_pop") * pl.col("surprisal_pop")).sum(),
            pop_meansq_seen=(pl.col("p_pop") * pl.col("surprisal_pop") ** 2).sum(),
            w_pop=pl.col("w_pop").first()))

    log_w_pop = pl.col("w_pop").log()
    pop_mass_unseen = 1 - pl.col("pop_mass_seen")   # p_pop mass on unseen values
    pop_mean_unseen = pop_mean_surprisal - pl.col("pop_mean_seen")
    pop_meansq_unseen = pop_meansq_surprisal - pl.col("pop_meansq_seen")

    # unseen surprisal = surprisal_pop - log(w_pop); expand the mean and
    # mean-square of that shifted variable over the unseen block, scaled by w_pop
    mean_surprisal = pl.col("seen_mean") + pl.col("w_pop") * (
        -log_w_pop * pop_mass_unseen + pop_mean_unseen)
    meansq_surprisal = pl.col("seen_meansq") + pl.col("w_pop") * (
        log_w_pop ** 2 * pop_mass_unseen
        - 2 * log_w_pop * pop_mean_unseen
        + pop_meansq_unseen)
    return per_user.with_columns(
        mean_surprisal=mean_surprisal, meansq_surprisal=meansq_surprisal
    ).select(USER, "mean_surprisal", "meansq_surprisal")


def fit_moments_logz(ct, hyper=Hyperparams()):
    """Per-user temperature from the fitted model's log Z (no resampling).

    center = mean surprisal (internal energy / entropy), scale = sqrt(variance of
    surprisal) (the heat capacity), both summed over the edge and auth axes,
    which the model treats as independent factors so their cumulants add. Gives a
    well-defined per-user temperature even for users with tied or single-event
    histories, where the MAD/std sample scale collapses to zero. Absent-from-train
    users fall back to the population moments, which this same formula returns at
    N_u=0.
    """
    pop_edge, edge_pop_mean, edge_pop_meansq = _pop_axis_moments(
        ct.pop_edge, "n", ct.N, ct.E, hyper.alpha_edge)
    pop_auth, auth_pop_mean, auth_pop_meansq = _pop_axis_moments(
        ct.pop_auth, "n", ct.N, ct.A, hyper.alpha_auth)
    edge = _user_axis_moments(ct.user_edge, pop_edge, EDGE_KEYS, "n",
                              ct.user_totals, hyper.kappa_edge,
                              edge_pop_mean, edge_pop_meansq)
    auth = _user_axis_moments(ct.user_auth, pop_auth, ("auth_type",), "n",
                              ct.user_totals, hyper.kappa_auth,
                              auth_pop_mean, auth_pop_meansq)

    # per-axis variance = meansq - mean^2; center and variance add across axes
    edge = edge.rename({"mean_surprisal": "edge_mean",
                        "meansq_surprisal": "edge_meansq"})
    auth = auth.rename({"mean_surprisal": "auth_mean",
                        "meansq_surprisal": "auth_meansq"})
    per_user = (
        edge.join(auth, on=USER, how="full", coalesce=True)
        .with_columns(
            center=pl.col("edge_mean") + pl.col("auth_mean"),
            variance=(pl.col("edge_meansq") - pl.col("edge_mean") ** 2)
            + (pl.col("auth_meansq") - pl.col("auth_mean") ** 2))
        .with_columns(scale=pl.col("variance").clip(lower_bound=0.0) ** 0.5)
        .select(USER, "center", "scale")
        .collect())

    pop_center = edge_pop_mean + auth_pop_mean
    pop_variance = ((edge_pop_meansq - edge_pop_mean ** 2)
                    + (auth_pop_meansq - auth_pop_mean ** 2))
    return Moments(per_user, pop_center, max(pop_variance, 0.0) ** 0.5)


def fit_moments(train, ct, hyper=Hyperparams(), method="mad"):
    """Pass 2: per-user surprisal center/scale from the train window.

    method="logz" derives a per-user temperature from the fitted model's
    partition function (see fit_moments_logz); "mad"/"meanstd" estimate the
    center/scale empirically from the train surprisals.
    """
    if method == "logz":
        return fit_moments_logz(ct, hyper)
    scored = event_surprisal(train, ct, hyper)
    center, scale = _center_scale(method)
    per_user = scored.group_by(USER).agg(center, scale).collect()
    pooled = scored.select(center, scale).collect()
    return Moments(per_user, pooled["center"][0], pooled["scale"][0])


def standardize(scored, moments, eps=1e-9):
    """Add z = (s - center)/(scale + eps), each user against their own normal.

    Fall back to the pooled center for users with no train row. Scale is
    floored at the pooled scale. Small per-user scale -> near-tied surprisal
    set -> z inflated on any deviation.
    """
    return (scored.join(moments.per_user.lazy(), on=USER, how="left")
            .with_columns(center=pl.col("center").fill_null(moments.center),
                          scale=pl.col("scale").fill_null(moments.scale))
            .with_columns(scale=pl.when(pl.col("scale") >= moments.scale)
                          .then(pl.col("scale")).otherwise(moments.scale))
            .with_columns(z=(pl.col("s") - pl.col("center"))
                          / (pl.col("scale") + eps)))


def window_observable(scored, window_seconds=WINDOW_SECONDS, temperature=None,
                      observable="mean"):
    """Attach z_win: an observable over the trailing per-user window
    (t - window_seconds, t].

    observable="mean" (default): z_win is the window mean of z, ranking on mean
    surprisal not the sum, so a busy but routine account does not outscore a
    shorter run of anomalous events (see the README surprisal-scoring section).

    observable="var": z_win is the window variance of z, a heat-capacity analog.

    observable="edges": z_win is the count of distinct (src_comp, dst_comp,
    auth_type) the user touches in the trailing window. Does not read z at all.

    Finite temperature T > 0 (overrides observable): z_win is the soft-max free
    energy
        G_T = T * log( mean_i exp(z_i / T) )
    over the window's per-event z_i. T -> inf recovers the mean, T -> 0 the
    window max.

    Hypothesis: Moderate T weights the window's high-z events above the mean
    without collapsing to a single max, matching the lateral-movement signature:
    a few high-z logins inside otherwise routine traffic.

    Finding: G_T interpolates mean..max, so low T weights the heavy right tail
    in z (benign novelty spikes) and underperforms the mean; high T recovers the
    mean. Swept over T in 0.5..64, no temperature beats the mean by more than
    one event.

    Needs the z column from standardize(). Sorts by (user, time) so each user's
    window sees its events in order; `time` is epoch seconds.
    """
    win = f"{int(window_seconds)}i"
    scored = scored.sort([USER, "time"])
    if temperature is not None:
        T = float(temperature)
        if T <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Shift by the user's max z so exp() stays in (0, 1] and never
        # overflows. Subtracting and adding the shift back must happen in the
        # same .over(USER) context, or the two are different maxima and each
        # user's G_T carries a constant offset that scrambles the ranking.
        c = pl.col("z").max()
        ebar = (((pl.col("z") - c) / T).exp()
                .rolling_mean_by("time", window_size=win))
        return scored.with_columns(z_win=(c + T * ebar.log()).over(USER))
    if observable == "mean":
        return scored.with_columns(
            z_win=pl.col("z").rolling_mean_by("time", window_size=win).over(USER))
    if observable == "var":
        return scored.with_columns(
            z_win=pl.col("z").rolling_var_by("time", window_size=win)
            .over(USER).fill_null(0.0))
    if observable == "edges":
        counts = (scored.rolling(index_column="time", period=win, group_by=USER)
                  .agg(pl.struct(*EDGE_KEYS, "auth_type").n_unique().alias("z_win"))
                  .unique(subset=[USER, "time"], keep="first"))
        return scored.join(counts, on=[USER, "time"], how="left")
    raise ValueError(
        f"unknown observable {observable!r}, want 'mean', 'var', or 'edges'")


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
    parser.add_argument("--method", choices=("mad", "meanstd", "logz"),
                        default="mad",
                        help="per-user standardization estimator")
    args = parser.parse_args()

    events = pl.scan_csv(args.input).select(NEEDED)
    if args.split_fraction is None:
        split = FIRST_REDTEAM_TIME
    else:
        split = split_time_from_fraction(events, args.split_fraction)

    # train/test leak guard
    train, test = split_frame(events, split)

    ct = count_tables(train)
    moments = fit_moments(train, ct, method=args.method)
    scored = standardize(event_surprisal(test, ct), moments)
    # z -> recall@budget score column
    ranked = (scored.with_columns(score=pl.col("z"))
              .sort(["score", "time"], descending=[True, False])
              .collect())

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surprisal_alerts.csv"
    ranked.write_csv(out_path)

    redteam = int((ranked["label"] == 1).sum())
    print(f"test {ranked.height} rows scored; {redteam} red team")
    print(f"ranked alerts written to {out_path}")

    print(f"\nper-day recall@budget (denominator: {redteam} red team)")
    with pl.Config(tbl_hide_dataframe_shape=True):
        print(recall_at_budget(ranked, redteam))

    print("\ntop alerts:")
    top = ranked.head(10).select("time", "src_user", "src_comp", "dst_comp",
                                 "auth_type", "s", "score", "label")
    with pl.Config(tbl_rows=10, tbl_hide_dataframe_shape=True):
        print(top)
