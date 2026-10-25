#!/usr/bin/env python3
"""Recall-vs-budget plot for every detector (baselines + both surprisal configs)
on the LANL test window.

Sweeps per-day alert budget and plots recall% for every method on one axis.
The surprisal/iforest CSVs are ~19M rows. Only time/score/label are projected
before collect -> eager frame stays small.

Writes outputs/recall_vs_budget.png.
"""
import polars as pl

import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt

from evaluate import recall_at_budget

# Test-window red team count, fixed across methods (see README Results). Split
# is the first red team event, FIRST_REDTEAM_TIME=150885 in src/timesplit.py.
TOTAL_REDTEAM = 702

# Budget sweep for x axis: per-day review budget, 10..100 alerts/day.
BUDGETS = list(range(10, 101, 5))

OUT_PNG = "outputs/recall_vs_budget.png"

# (label, csv, hex). Colors are colorblind-safe on a light surface.
METHODS = [
    ("Surprisal, windowed mean", "outputs/surprisal_mean_alerts.csv", "#2a78d6"),
    ("Surprisal, per-event", "outputs/surprisal_alerts.csv", "#7b3fa0"),
    ("Isolation Forest", "outputs/iforest_alerts.csv", "#eb6834"),
    ("Novelty edge baseline", "outputs/novelty_alerts.csv", "#008300"),
]


def load_scored(path):
    return pl.scan_csv(path).select("time", "score", "label").collect()


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, path, color in METHODS:
        scored = load_scored(path)
        table = recall_at_budget(scored, TOTAL_REDTEAM, budgets=BUDGETS)
        recall_pct = (table["recall"] * 100).to_list()
        ax.plot(table["budget"].to_list(), recall_pct,
                label=label, color=color, linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Alert budget (top-B scored events reviewed per day)")
    ax.set_ylabel("Red-team recall (%)")
    ax.set_title(f"Recall vs alert budget on LANL test window "
                 f"({TOTAL_REDTEAM} red-team events)")
    ax.set_xlim(BUDGETS[0], BUDGETS[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
