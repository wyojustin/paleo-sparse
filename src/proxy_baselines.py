"""
No-learning baselines with leave-one-site-out validation on PAGES 2k proxies.
Measurement-only track: no climate model, no reanalysis.

Data
  Annual-resolution records (median spacing 1 yr), annual matrix restricted
  to a window [y0, y1), each record z-scored over its own values in the
  window. Records with fewer than MIN_YEARS values in the window are
  dropped (reported). Records sharing coordinates are averaged into one
  site series S[t, s] (T x Ns): leaving out a site leaves out all its
  records, so no target is predicted from its own co-located twin.

Geometry (sphere, unit vectors u_s in R^3)
  nearest   zhat_s(t) = S[t, j*],  j* = argmin_{j != s, S[t, j] observed} angle(u_s, u_j)
  delaunay  spherical Delaunay of the observed sites (j != s) in year t is the
            convex hull of their unit vectors. Find the facet (a, b, c) whose
            cone contains u_s: u_s = lam_a u_a + lam_b u_b + lam_c u_c, lam >= 0.
            zhat_s(t) = sum_i (lam_i / sum lam) S[t, i]   (barycentric, linear)
  Triangulations depend on which sites are observed; they are cached per
  availability pattern.

Leave-one-out score for site s over years where it is observed and a
prediction exists:
  r_s  = corr_t(z, zhat)
  SS_s = 1 - mean (z - zhat)^2 / mean (z - mean z)^2   (0 = predict the mean)
  Note for copying a noisy neighbor: if both have unit variance and
  correlation r, then MSE = 2 (1 - r), so SS = 2r - 1 < 0 unless r > 0.5.
  We also report SS of the best scaling b zhat (b = r * sd z / sd zhat),
  SS_opt = r^2: the skill a calibrated linear rescaling would reach.
  SS_opt uses the target itself and is a diagnostic, not a method.

Usage
  python src/proxy_baselines.py --window 1000 2001 --window 1500 2001
  python src/proxy_baselines.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import os
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_data import annual_matrix, load_pages2k  # noqa: E402

MIN_YEARS = 50


def unit(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


# --------------------------------------------------------------------------
# data -> site matrix
# --------------------------------------------------------------------------
def site_matrix(recs, window, min_years=MIN_YEARS, log=print):
    """Annual records in window, z-scored, averaged by site.
    Returns S (T, Ns), years, site dict list, report counter."""
    rep = collections.Counter()
    ann = [r for r in recs if 0.99 < r.spacing <= 1.01]
    rep["annual-resolution records"] = len(ann)
    Y, years = annual_matrix(ann, span=window)
    n = np.isfinite(Y).sum(0)
    keep = n >= min_years
    rep[f"dropped: < {min_years} values in window"] = int((~keep).sum())
    Y = Y[:, keep]
    ann = [r for r, k in zip(ann, keep) if k]
    Z = (Y - np.nanmean(Y, 0)) / np.nanstd(Y, 0, ddof=1)
    groups = collections.OrderedDict()
    for m, r in enumerate(ann):
        groups.setdefault((round(r.lat, 4), round(r.lon, 4)), []).append(m)
    S = np.full((years.size, len(groups)), np.nan)
    sites = []
    for s, ((la, lo), ms) in enumerate(groups.items()):
        if len(ms) > 1:                              # years with none of them stay NaN
            cnt = np.isfinite(Z[:, ms]).sum(1)
            tot = np.nansum(Z[:, ms], axis=1)
            S[:, s] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        else:
            S[:, s] = Z[:, ms[0]]
        sites.append(dict(lat=la, lon=lo, n_rec=len(ms),
                          archive="/".join(sorted({ann[m].archive for m in ms})),
                          proxy="/".join(sorted({ann[m].proxy for m in ms})),
                          ids=",".join(ann[m].pid for m in ms)))
    rep["records used"] = len(ann)
    rep["sites"] = len(sites)
    rep["sites with > 1 record (averaged)"] = sum(s["n_rec"] > 1 for s in sites)
    return S, years, sites, rep


# --------------------------------------------------------------------------
# predictors: each returns zhat (T,) for target s, NaN where impossible
# --------------------------------------------------------------------------
def predict_nearest(S, U, s):
    ang = np.arccos(np.clip(U @ U[s], -1, 1))
    ang[s] = np.inf
    D = np.where(np.isfinite(S), ang[None, :], np.inf)    # (T, Ns)
    j = np.argmin(D, axis=1)
    ok = np.isfinite(D[np.arange(S.shape[0]), j])
    zhat = np.where(ok, S[np.arange(S.shape[0]), j], np.nan)
    dist_km = np.where(ok, 6371.0 * ang[j], np.nan)
    return zhat, dist_km


def _facet_weights(P, u):
    """P (n, 3) unit vectors (hull of them), u (3,). Returns (idx (3,),
    weights (3,), max edge angle in rad) of the facet whose cone holds u,
    or None if no facet does (degenerate hull)."""
    hull = ConvexHull(P)
    F = hull.simplices                                    # (nf, 3)
    A = P[F].transpose(0, 2, 1)                           # (nf, 3, 3), columns = vertices
    with np.errstate(all="ignore"):
        lam = np.linalg.solve(A, np.broadcast_to(u, (F.shape[0], 3))[..., None])[..., 0]
    ok = np.all(lam >= -1e-12, axis=1) & np.all(np.isfinite(lam), axis=1)
    if not ok.any():
        return None
    f = np.where(ok)[0][np.argmax(lam[ok].sum(1))]        # the nearest facet plane
    w = lam[f] / lam[f].sum()
    V = P[F[f]]
    edge = max(np.arccos(np.clip(V[i] @ V[(i + 1) % 3], -1, 1)) for i in range(3))
    return F[f], w, edge


def predict_delaunay(S, U, s, cache=None):
    T = S.shape[0]
    zhat = np.full(T, np.nan)
    edge_km = np.full(T, np.nan)
    avail = np.isfinite(S)
    avail[:, s] = False
    need = np.where(np.isfinite(S[:, s]))[0]
    cache = {} if cache is None else cache
    for t in need:
        idx = np.where(avail[t])[0]
        if idx.size < 4:
            continue
        key = (s, avail[t].tobytes())
        if key not in cache:
            res = _facet_weights(U[idx], U[s])
            cache[key] = None if res is None else (idx[res[0]], res[1], res[2])
        res = cache[key]
        if res is None:
            continue
        vi, w, e = res
        zhat[t] = w @ S[t, vi]
        edge_km[t] = 6371.0 * e
    return zhat, edge_km


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(z, zhat, min_n=20):
    ok = np.isfinite(z) & np.isfinite(zhat)
    n = int(ok.sum())
    if n < min_n:
        return dict(n=n, r=np.nan, ss=np.nan, ss_opt=np.nan, rmse=np.nan)
    a, b = z[ok], zhat[ok]
    va = np.mean((a - a.mean()) ** 2)
    r = np.corrcoef(a, b)[0, 1] if b.std() > 0 else np.nan
    return dict(n=n, r=r, ss=1 - np.mean((a - b) ** 2) / va,
                ss_opt=r**2 if np.isfinite(r) else np.nan, rmse=np.sqrt(np.mean((a - b) ** 2)))


def loo(S, U, methods, log=print):
    """methods: {name: fn(S, U, s) -> (zhat, aux)}. Returns {name: list of
    per-site dicts}."""
    out = {m: [] for m in methods}
    for name, fn in methods.items():
        t0 = time.time()
        for s in range(S.shape[1]):
            zhat, aux = fn(S, U, s)
            d = score(S[:, s], zhat)
            d["aux_median_km"] = float(np.nanmedian(aux)) if np.isfinite(aux).any() else np.nan
            d["coverage"] = float(np.mean(np.isfinite(zhat[np.isfinite(S[:, s])])))
            out[name].append(d)
        log(f"    {name}: {time.time() - t0:.1f} s")
    return out


def summarize(res, sites, log=print):
    for name, rows in res.items():
        r = np.array([d["r"] for d in rows]); ss = np.array([d["ss"] for d in rows])
        so = np.array([d["ss_opt"] for d in rows]); cov = np.array([d["coverage"] for d in rows])
        ok = np.isfinite(ss)
        log(f"  {name:9s} sites scored {ok.sum():3d}/{len(rows)}  "
            f"median r {np.nanmedian(r):+.3f}  median SS {np.nanmedian(ss):+.3f}  "
            f"SS > 0: {np.mean(ss[ok] > 0):.0%}  median SS_opt (r^2) {np.nanmedian(so):.3f}  "
            f"coverage {np.nanmean(cov):.2f}  median {('NN distance' if name == 'nearest' else 'max triangle edge')} "
            f"{np.nanmedian([d['aux_median_km'] for d in rows]):.0f} km")


# --------------------------------------------------------------------------
# self-test: smooth field on the sphere, interpolation must work
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    Ns, T = 300, 60
    lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, Ns))); lon = rng.uniform(-180, 180, Ns)
    U = unit(lat, lon)
    # field = random linear combination of u_x, u_y, u_z (degree-1 harmonics):
    # Delaunay barycentric weights reproduce it up to the chord-vs-arc factor
    C = rng.standard_normal((T, 3))
    S = C @ U.T
    S[rng.random(S.shape) < 0.1] = np.nan                # 10 % missing
    res = loo(S, U, {"nearest": predict_nearest, "delaunay": predict_delaunay},
              log=lambda *a: None)
    med = {k: np.nanmedian([d["ss"] for d in v]) for k, v in res.items()}
    print(f"[selftest] smooth degree-1 field, {Ns} sites: median SS nearest "
          f"{med['nearest']:.3f}, delaunay {med['delaunay']:.3f}")
    assert med["delaunay"] > 0.99 and med["delaunay"] > med["nearest"] > 0.8, med
    # pure noise: no skill, SS ~ -1 for nearest (copying independent noise)
    Sn = rng.standard_normal((200, Ns))
    rn = loo(Sn, U, {"nearest": predict_nearest}, log=lambda *a: None)
    m = np.nanmedian([d["ss"] for d in rn["nearest"]])
    print(f"[selftest] white noise: median SS nearest {m:.2f} (theory -1)")
    assert -1.3 < m < -0.7, m
    print("[selftest] PASS")


# --------------------------------------------------------------------------
def main(path, windows, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs, rep0 = load_pages2k(path)
    date = _dt.date.today().isoformat()
    os.makedirs("figures", exist_ok=True)
    ink, muted = "#222222", "#777777"
    cN, cD = "#2a6fdb", "#d9480f"
    fig = plt.figure(figsize=(13, 4.6 * len(windows)), layout="constrained")
    sfs = fig.subfigures(len(windows), 1) if len(windows) > 1 else [fig]
    rows_csv = []
    for wi, window in enumerate(windows):
        print(f"\nWindow [{window[0]}, {window[1]})")
        S, years, sites, rep = site_matrix(recs, window)
        for k, v in rep.items():
            print(f"  {k}: {v}")
        print(f"  site matrix S: {S.shape} (T x Ns), filled {np.isfinite(S).mean():.3f}")
        U = unit(np.array([s["lat"] for s in sites]), np.array([s["lon"] for s in sites]))
        cache = {}
        res = loo(S, U, {"nearest": predict_nearest,
                         "delaunay": lambda S_, U_, s: predict_delaunay(S_, U_, s, cache)})
        summarize(res, sites)
        # by archive
        arch = collections.Counter(s["archive"] for s in sites)
        print("  median SS / median r by site archive (sites):")
        for a, n in arch.most_common():
            ix = [i for i, s in enumerate(sites) if s["archive"] == a]
            msg = "    " + f"{a:28s} ({n:3d})"
            for name in res:
                msg += f"  {name} {np.nanmedian([res[name][i]['ss'] for i in ix]):+.2f} / " \
                       f"{np.nanmedian([res[name][i]['r'] for i in ix]):+.2f}"
            print(msg)
        for i, s in enumerate(sites):
            for name in res:
                rows_csv.append(dict(window=f"[{window[0]}, {window[1]})", method=name, **s,
                                     **{k: res[name][i][k] for k in ("n", "r", "ss", "ss_opt", "rmse",
                                                                     "coverage", "aux_median_km")}))

        # ---------- figure row ----------
        sf = sfs[wi]
        ax = sf.add_subplot(1, 3, 1)
        ssN = np.array([d["ss"] for d in res["nearest"]]); ssD = np.array([d["ss"] for d in res["delaunay"]])
        ax.scatter(ssN, ssD, s=10, color=ink, lw=0, alpha=0.7)
        lo = np.nanmin(np.r_[ssN, ssD]); ax.plot([lo, 1], [lo, 1], color=muted, lw=0.8, ls="--")
        ax.axhline(0, color=muted, lw=0.6); ax.axvline(0, color=muted, lw=0.6)
        ax.set_xlabel("SS, nearest neighbor", color=muted); ax.set_ylabel("SS, Delaunay", color=muted)
        ax.set_title(f"[{window[0]}, {window[1]}): per-site skill, {len(sites)} sites", color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)

        ax = sf.add_subplot(1, 3, 2)
        dN = np.array([d["aux_median_km"] for d in res["nearest"]])
        rN = np.array([d["r"] for d in res["nearest"]]); rD = np.array([d["r"] for d in res["delaunay"]])
        ax.scatter(dN, rN, s=10, color=cN, lw=0, alpha=0.7, label="nearest: r")
        ax.scatter(dN, rD, s=10, color=cD, lw=0, alpha=0.7, label="Delaunay: r")
        ax.set_xscale("log"); ax.axhline(0, color=muted, lw=0.6)
        ax.axhline(0.5, color=muted, lw=0.6, ls=":")
        ax.text(np.nanmin(dN[dN > 0]), 0.52, "r = 0.5: SS = 0 for a copied neighbor",
                color=muted, fontsize=7)
        ax.set_xlabel("distance to nearest observed site (km, median over years)", color=muted)
        ax.set_ylabel("correlation r with held-out site", color=muted)
        ax.set_title("Skill vs distance", color=ink, fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, lw=0.5, which="both")

        ax = sf.add_subplot(1, 3, 3)
        order = [a for a, _ in arch.most_common()][:8]
        rng = np.random.default_rng(0)
        for i, a in enumerate(order):
            ix = [k for k, s in enumerate(sites) if s["archive"] == a]
            for name, c, off in (("nearest", cN, -0.15), ("delaunay", cD, 0.15)):
                v = np.array([res[name][k]["r"] for k in ix])
                ax.scatter(v, i + off + rng.uniform(-0.08, 0.08, v.size), s=8, color=c, lw=0, alpha=0.7)
                ax.plot([np.nanmedian(v)] * 2, [i + off - 0.12, i + off + 0.12], color=c, lw=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{a} ({arch[a]})" for a in order], fontsize=8)
        ax.invert_yaxis(); ax.axvline(0, color=muted, lw=0.6)
        ax.set_xlabel("r (bar = median; blue NN, orange Delaunay)", color=muted)
        ax.set_title("By site archive", color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)

    stem = f"figures/{date}_proxy_baselines_loo"
    with open(stem + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wr.writeheader(); wr.writerows(rows_csv)
    fig.savefig(stem + ".png", dpi=130)
    print(f"\nwrote {stem}.csv\nwrote {stem}.png")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/cfr-data/pages2kv2.json")
    ap.add_argument("--window", type=int, nargs=2, action="append",
                    help="half-open [y0, y1); repeat for several")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.path, [tuple(w) for w in (a.window or [[1000, 2001], [1500, 2001]])],
             show=not a.no_show)
