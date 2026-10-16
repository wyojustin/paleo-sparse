"""
PAGES 2k v2 proxy records (cfr-data pages2kv2.json) -> annual matrix Y.

Measurement-only track: no climate model, no reanalysis as truth.

Conventions
  * Row-vector: Y[t, m] is year t, record m (T x M), NaN where missing.
  * Half-open ranges [a, b). Calendar year y holds samples with
    floor(year) == y, i.e. year values in [y, y + 1).
  * Never fail silently: problems raise named errors; every repair
    (dropped NaN pairs, re-sorting, duplicate years) is counted and reported.

Steps
  1. Parse each record: drop pairs whose year or value is missing ('nan'),
     sort by year (191 records are stored newest first), keep duplicates.
  2. Resolution of a record: median positive spacing of its sample years.
  3. Annual matrix: Y[t, m] = mean of record m's samples in [y_t, y_t + 1).
     A coarse record (e.g. one sample per decade) fills one year per sample;
     the gaps stay NaN (no interpolation).
  4. Sub-annual records: seasonal structure. With n = round(1 / spacing)
     samples per year, phase bin b = floor(n * frac(year)). Seasonal
     variance fraction eta^2 = var(bin means, sample weighted) / var(values).
  5. Normalization (for later steps): z-score each record over its own
     annual values, z = (Y - mean) / sd.

Usage
  python src/proxy_data.py [data/cfr-data/pages2kv2.json] [--no-show]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class Record:
    pid: str
    archive: str
    proxy: str
    variable: str
    units: str
    lat: float
    lon: float
    year: np.ndarray        # (n,) ascending, fractional allowed
    value: np.ndarray       # (n,)

    @property
    def spacing(self) -> float:
        d = np.diff(np.unique(self.year))
        return float(np.median(d)) if d.size else np.nan


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and x.strip().lower() == "nan":
        return np.nan
    raise ValueError(f"unexpected entry {x!r} (type {type(x).__name__})")


NEED = ("paleoData_pages2kID", "archiveType", "paleoData_proxy", "paleoData_variableName",
        "paleoData_units", "geo_meanLat", "geo_meanLon", "year", "yearUnits",
        "paleoData_values")


def load_pages2k(path):
    """Return (records, report). report counts every repair made."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"proxy file not found: {path}  (see README: cfr-data)")
    with open(path) as f:
        d = json.load(f)
    missing = [k for k in NEED if k not in d]
    if missing:
        raise KeyError(f"{path} lacks fields {missing}; has {list(d)}")
    rep = collections.Counter()
    recs = []
    for k in sorted(d["year"], key=int):
        pid = d["paleoData_pages2kID"][k]
        if d["yearUnits"][k] != "AD":
            raise ValueError(f"{pid}: yearUnits {d['yearUnits'][k]!r}, only 'AD' handled")
        y = np.array([_num(x) for x in d["year"][k]])
        v = np.array([_num(x) for x in d["paleoData_values"][k]])
        if y.size != v.size:
            raise ValueError(f"{pid}: {y.size} years but {v.size} values")
        ok = np.isfinite(y) & np.isfinite(v)
        if (~ok).any():
            rep["records with dropped NaN pairs"] += 1
            rep["NaN pairs dropped"] += int((~ok).sum())
        y, v = y[ok], v[ok]
        if y.size < 2:
            raise ValueError(f"{pid}: fewer than 2 valid samples")
        dy = np.diff(y)
        if np.all(dy < 0):
            rep["records stored newest first (reversed)"] += 1
        elif np.any(dy < 0):
            rep["records in mixed order (sorted)"] += 1
        o = np.argsort(y, kind="stable")
        y, v = y[o], v[o]
        if np.any(np.diff(y) == 0):
            rep["records with duplicate years (kept, averaged when binned)"] += 1
        lat, lon = float(d["geo_meanLat"][k]), float(d["geo_meanLon"][k])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ValueError(f"{pid}: missing lat/lon")
        recs.append(Record(pid, d["archiveType"][k], d["paleoData_proxy"][k],
                           d["paleoData_variableName"][k], str(d["paleoData_units"][k]),
                           lat, lon, y, v))
    return recs, rep


def annual_matrix(recs, span=None):
    """Y (T x M) of annual means, years (T,). span: optional half-open
    [y0, y1); default covers every sample."""
    if span is None:
        y0 = int(np.floor(min(r.year[0] for r in recs)))
        y1 = int(np.floor(max(r.year[-1] for r in recs))) + 1
    else:
        y0, y1 = span
    years = np.arange(y0, y1)
    Y = np.full((years.size, len(recs)), np.nan)
    for m, r in enumerate(recs):
        t = np.floor(r.year).astype(int) - y0
        ok = (t >= 0) & (t < years.size)
        s = np.bincount(t[ok], weights=r.value[ok], minlength=years.size)
        n = np.bincount(t[ok], minlength=years.size)
        Y[n > 0, m] = s[n > 0] / n[n > 0]
    return Y, years


def zscore(Y):
    """Per-record z-score over its own non-missing annual values."""
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1)
    if np.any(~np.isfinite(sd) | (sd == 0)):
        bad = np.where(~np.isfinite(sd) | (sd == 0))[0]
        raise ValueError(f"records {bad[:10]} have zero or undefined variance")
    return (Y - mu) / sd, mu, sd


def seasonal(r: Record):
    """For a sub-annual record: (n bins per year, eta^2, seasonal amplitude /
    interannual sd of annual means)."""
    n = int(round(1.0 / r.spacing))
    if n < 2:
        raise ValueError(f"{r.pid}: not sub-annual (spacing {r.spacing:.3f})")
    frac = r.year - np.floor(r.year)
    b = np.minimum(np.floor(n * frac + 1e-9).astype(int), n - 1)
    cnt = np.bincount(b, minlength=n)
    mean_b = np.bincount(b, weights=r.value, minlength=n) / np.maximum(cnt, 1)
    eta2 = np.sum(cnt * (mean_b - r.value.mean()) ** 2) / np.sum((r.value - r.value.mean()) ** 2)
    yr = np.floor(r.year).astype(int)
    full = np.array([y for y in np.unique(yr) if np.sum(yr == y) == n])
    if full.size < 3:
        return n, eta2, np.nan
    ann = np.array([r.value[yr == y].mean() for y in full])
    amp = mean_b[cnt > 0].max() - mean_b[cnt > 0].min()
    return n, eta2, amp / ann.std(ddof=1)


# --------------------------------------------------------------------------
def main(path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs, rep = load_pages2k(path)
    M = len(recs)
    print(f"PAGES 2k v2: {M} records from {os.path.basename(path)}")
    for k, v in rep.items():
        print(f"  repair: {k}: {v}")

    arch = collections.Counter(r.archive for r in recs)
    print("\nArchives: " + ", ".join(f"{a} {n}" for a, n in arch.most_common()))
    sites = {(r.lat, r.lon) for r in recs}
    print(f"Distinct site coordinates: {len(sites)} ({M - len(sites)} records share a site)")
    print(f"Hemispheres: {sum(r.lat >= 0 for r in recs)} N, {sum(r.lat < 0 for r in recs)} S")

    # time coverage
    first = np.array([r.year[0] for r in recs]); last = np.array([r.year[-1] for r in recs])
    span_len = last - first
    print(f"\nTime coverage: earliest sample {first.min():.1f}, latest {last.max():.1f}")
    print(f"  record length (last - first, yr): median {np.median(span_len):.0f}, "
          f"[{span_len.min():.0f}, {span_len.max():.0f}]")
    for y in (1, 500, 1000, 1500, 1850):
        print(f"  records starting before {y:4d}: {np.sum(first < y)}")

    # resolution
    sp = np.array([r.spacing for r in recs])
    edges = [(0, 0.99, "sub-annual (< 1 yr)"), (0.99, 1.01, "annual"),
             (1.01, 5, "(1, 5] yr"), (5, 20, "(5, 20] yr"), (20, np.inf, "> 20 yr")]
    print("\nResolution (median sample spacing):")
    for a, b, lab in edges:
        sel = (sp > a) & (sp <= b) if a else (sp <= b)
        by = collections.Counter(recs[i].archive for i in np.where(sel)[0])
        print(f"  {lab:20s} {sel.sum():4d}   " + ", ".join(f"{k} {v}" for k, v in by.most_common()))

    # annual matrix and missingness
    Y, years = annual_matrix(recs)
    print(f"\nAnnual matrix Y: {Y.shape} (T x M), years [{years[0]}, {years[-1] + 1})")
    print(f"  filled fraction, whole matrix: {np.isfinite(Y).mean():.3f}")
    inside = np.zeros_like(Y, bool)
    for m, r in enumerate(recs):
        inside[int(np.floor(r.year[0])) - years[0]:int(np.floor(r.year[-1])) - years[0] + 1, m] = True
    print(f"  filled fraction inside each record's own span: {np.isfinite(Y)[inside].mean():.3f} "
          f"(gaps come from coarse resolution, not dropouts)")
    for a, b in ((1, 2001), (1000, 2001), (1500, 2001), (1850, 2001), (1951, 1981)):
        sl = slice(a - years[0], b - years[0])
        f_ = np.isfinite(Y[sl])
        print(f"  [{a}, {b}): filled {f_.mean():.3f}; records with any data {np.sum(f_.any(0))}; "
              f"records >= 90 % complete {np.sum(f_.mean(0) >= 0.9)}")

    # sub-annual: seasonal structure
    sub = [r for r in recs if r.spacing < 0.99]
    print(f"\nSub-annual records: {len(sub)}")
    seas = []
    for r in sub:
        n, e2, ratio = seasonal(r)
        seas.append((r, n, e2, ratio))
    if seas:
        e2s = np.array([s[2] for s in seas]); rat = np.array([s[3] for s in seas])
        nb = collections.Counter(s[1] for s in seas)
        print(f"  samples per year: " + ", ".join(f"{k}/yr: {v}" for k, v in sorted(nb.items())))
        print(f"  seasonal variance fraction eta^2: median {np.median(e2s):.2f}, "
              f"[{e2s.min():.2f}, {e2s.max():.2f}]")
        print(f"  seasonal amplitude / interannual sd: median {np.nanmedian(rat):.1f}")
        by = collections.Counter(f"{s[0].archive} {s[0].proxy}" for s in seas)
        print("  by archive/proxy: " + ", ".join(f"{k} {v}" for k, v in by.most_common()))

    Z, mu, sd = zscore(Y)
    print(f"\nz-scored Y ready (per record, own annual values); mean |mu| check "
          f"{np.nanmean(np.abs(np.nanmean(Z, 0))):.1e}")

    # ---------------- figure ----------------
    ink, muted = "#222222", "#777777"
    pal = ["#2a6fdb", "#d9480f", "#2f9e44", "#7048e8", "#c2255c", "#e8a33d", "#0c8599"]
    order = [a for a, _ in arch.most_common()]
    col = {a: (pal[i] if i < len(pal) else "#999999") for i, a in enumerate(order)}
    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    for a in order:
        rr = [r for r in recs if r.archive == a]
        ax.scatter([r.lon for r in rr], [r.lat for r in rr], s=10, color=col[a],
                   label=f"{a} ({len(rr)})" if col[a] != "#999999" else None, lw=0)
    ax.scatter([], [], color="#999999", s=10, label="other")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_title(f"{M} records at {len(sites)} sites", color=ink, fontsize=10)
    ax.set_xlabel("lon", color=muted); ax.set_ylabel("lat", color=muted)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[0, 1])
    avail = np.isfinite(Y)
    span_av = inside
    for a in order:
        ms = [m for m, r in enumerate(recs) if r.archive == a]
        if col[a] == "#999999":
            continue
        ax.plot(years, span_av[:, ms].sum(1), color=col[a], lw=1.2, label=a)
    ax.plot(years, span_av.sum(1), color=ink, lw=1.6, label="all (within span)")
    ax.plot(years, avail.sum(1), color=ink, lw=0.8, ls=":", label="all (sample in that year)")
    ax.set_yscale("log"); ax.set_xlim(0, 2020)
    ax.set_title("Records available per year", color=ink, fontsize=10)
    ax.set_xlabel("year CE", color=muted); ax.set_ylabel("records", color=muted)
    ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.25, lw=0.5, which="both")

    ax = fig.add_subplot(gs[1, 0])                 # dot plot, jittered by archive
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        ms = [m for m, r in enumerate(recs) if r.archive == a]
        ax.scatter(sp[ms], i + rng.uniform(-0.25, 0.25, len(ms)), s=10,
                   color=col[a] if col[a] != "#999999" else "#999999", lw=0, alpha=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_xscale("log"); ax.axvline(1, color=muted, lw=0.8, ls="--")
    ax.set_xlabel("median sample spacing (yr)", color=muted)
    ax.set_title("Resolution per record", color=ink, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which="both"); ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 1])
    if seas:                                       # strip plot by archive / proxy
        groups = collections.defaultdict(list)
        for r, n, e2, _ in seas:
            groups[f"{r.archive} {r.proxy}"].append(e2)
        gl = sorted(groups, key=lambda g: -len(groups[g]))
        for i, g in enumerate(gl):
            v = np.array(groups[g])
            ax.scatter(v, i + rng.uniform(-0.2, 0.2, v.size), s=14, color="#2a6fdb", lw=0)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=ink, lw=1.5)
        ax.set_yticks(range(len(gl)))
        ax.set_yticklabels([f"{g} ({len(groups[g])})" for g in gl], fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_xlabel("seasonal variance fraction eta^2 (bar = median)", color=muted)
        ax.set_title(f"Annual-cycle structure, {len(seas)} sub-annual records",
                     color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("PAGES 2k v2 proxy network (measurements only)", color=ink, fontsize=11)
    os.makedirs("figures", exist_ok=True)
    png = f"figures/{_dt.date.today().isoformat()}_proxy_data_overview.png"
    fig.savefig(png, dpi=130)
    print(f"\nwrote {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/cfr-data/pages2kv2.json")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    main(a.path, show=not a.no_show)
