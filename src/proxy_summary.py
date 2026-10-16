"""
Summary figure for the leave-one-site-out architecture survey: per-site SS
of every method (dot plot, bar = median), and per-site paired differences
against kriging, for each window in results/scores_*.csv.

Usage
  python src/proxy_summary.py [--no-show]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import os

import numpy as np

METHODS = ["nearest", "delaunay", "ridge", "krige", "mlp", "lstm", "tcn"]


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    sites = sorted({int(r["site"]) for r in rows})
    ss = {m: np.full(len(sites), np.nan) for m in METHODS}
    tree = np.zeros(len(sites), bool)
    pos = {s: i for i, s in enumerate(sites)}
    for r in rows:
        if r["method"] not in ss:
            raise KeyError(f"{path}: unknown method {r['method']}")
        i = pos[int(r["site"])]
        ss[r["method"]][i] = float(r["ss"]) if r["ss"] not in ("", "nan") else np.nan
        tree[i] = r["archive"] == "tree"
    return ss, tree


def main(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob("results/scores_*.csv"))
    if not files:
        raise FileNotFoundError("no results/scores_*.csv; run src/proxy_learned.py first")
    ink, muted, cT, cO = "#222222", "#777777", "#2f9e44", "#d9480f"
    fig, axs = plt.subplots(len(files), 2, figsize=(13, 4.3 * len(files)), layout="constrained",
                            squeeze=False)
    rng = np.random.default_rng(0)
    for k, f in enumerate(files):
        w = os.path.basename(f)[7:-4].split("_")
        ss, tree = load(f)
        ax = axs[k, 0]
        for i, m in enumerate(METHODS):
            v = ss[m]
            y = i + rng.uniform(-0.25, 0.25, v.size)
            ax.scatter(np.clip(v[tree], -2, 1), y[tree], s=5, color=cT, lw=0, alpha=0.5)
            ax.scatter(np.clip(v[~tree], -2, 1), y[~tree], s=5, color=cO, lw=0, alpha=0.6)
            ax.plot([np.nanmedian(v)] * 2, [i - 0.35, i + 0.35], color=ink, lw=2)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS); ax.invert_yaxis()
        ax.axvline(0, color=muted, lw=0.8, ls="--")
        ax.set_xlim(-2.05, 1)
        ax.set_xlabel("leave-one-site-out SS (clipped at -2; bar = median)", color=muted)
        ax.set_title(f"[{w[0]}, {w[1]}): per-site skill, green tree / orange other sites",
                     color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax = axs[k, 1]
        others = [m for m in METHODS if m != "krige"]
        for i, m in enumerate(others):
            dlt = ss[m] - ss["krige"]
            y = i + rng.uniform(-0.25, 0.25, dlt.size)
            ax.scatter(np.clip(dlt[tree], -1, 0.3), y[tree], s=5, color=cT, lw=0, alpha=0.5)
            ax.scatter(np.clip(dlt[~tree], -1, 0.3), y[~tree], s=5, color=cO, lw=0, alpha=0.6)
            ax.plot([np.nanmedian(dlt)] * 2, [i - 0.35, i + 0.35], color=ink, lw=2)
            ax.text(0.31, i, f" better than krige at {np.nanmean(dlt > 0):.0%}", va="center",
                    fontsize=7, color=muted)
        ax.set_yticks(range(len(others))); ax.set_yticklabels(others); ax.invert_yaxis()
        ax.axvline(0, color=muted, lw=0.8, ls="--")
        ax.set_xlim(-1.05, 0.8)
        ax.set_xlabel("SS(method) - SS(krige), per site (clipped)", color=muted)
        ax.set_title("Paired against kriging", color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    png = f"figures/{_dt.date.today().isoformat()}_proxy_architecture_survey.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-show", action="store_true")
    main(show=not ap.parse_args().no_show)
