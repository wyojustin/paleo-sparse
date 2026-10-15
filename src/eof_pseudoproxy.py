"""
EOFs, pseudoproxies and an EOF/l2 baseline reconstruction for pseudoproxy
experiments (PPEs).

Conventions
  * X is (T, N): one row per year, one column per grid cell (row-vector).
  * Half-open index/year ranges [a, b), like Python slices.
  * Never fail silently: problems raise named errors; recoverable gaps ask
    quit/continue when interactive, abort otherwise, and are summarized.

Math (row-vector form)
  Calibration rows c = [c0, c1). mu = mean_c(X), Z = (X_c - mu) * w,
  w = sqrt(cos(lat)) so that Z Z^T is an area-weighted covariance.
      Z = U S V^T,   E = V_r^T  (r x N EOFs),   PC_c = Z E^T = U_r S_r  (Tc x r)
  Proxies P (T x M), centered with their calibration mean.
  Inverse model (proxies regressed onto PCs):
      P_c = PC_c A + R,          A = argmin ||P_c - PC_c A||_F  (r x M)
      D   = diag(1 / var(R))     (GLS: noisy proxies count less)
  Reconstruction at any time t (ridge on PC amplitudes, lam >= 0):
      pc_hat = p_t D A^T (A D A^T + lam I)^{-1}
      x_hat  = (pc_hat E) / w + mu
  Metrics on validation rows v = [v0, v1):
      corr_j = corr_t(x_tj, xhat_tj)
      CE_j   = 1 - sum_t (x_tj - xhat_tj)^2 / sum_t (x_tj - mean_v x_j)^2
      GMT    = cos(lat)-weighted mean over cells (area weight, not sqrt).

Usage
  python src/eof_pseudoproxy.py                        # random proxy sites
  python src/eof_pseudoproxy.py --pages2k pages2kv2.json  # real PAGES 2k sites
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------
# loud-failure helper (same contract as the loader)
# --------------------------------------------------------------------------
def ask_continue(msg: str) -> None:
    if sys.stdin is not None and sys.stdin.isatty():
        ans = input(f"\nWARNING: {msg}\n  [q]uit or [c]ontinue? ").strip().lower()
        if ans.startswith("c"):
            return
        raise SystemExit(f"Aborted by user: {msg}")
    raise RuntimeError(f"Non-interactive run, aborting: {msg}")


def _check_X(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T x N), got shape {X.shape}")
    bad = ~np.isfinite(X)
    if bad.any():
        raise ValueError(f"X has {bad.sum()} non-finite entries; mask them in the loader")
    return X


_NOTED = {}


def safe_weights(w):
    """sqrt(cos(lat)) is 0 at the poles (CESM f19 has lat = +/-90). Dividing
    by 0 on the way back is undefined, so floor w at the smallest positive
    value and say so."""
    w = np.asarray(w, dtype=np.float64)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and >= 0")
    z = w < 1e-6 * w.max()          # cos(90 deg) is ~6e-17 in floating point, not 0
    if z.any():
        floor = w[~z].min()
        if not _NOTED.get("poles"):
            _NOTED["poles"] = True
            print(f"[safe_weights] NOTE: {z.sum()} cells with w = 0 (poles) floored to "
                  f"{floor:.3g} (reported once)", file=sys.stderr)
        w = np.where(z, floor, w)
    return w


# --------------------------------------------------------------------------
# EOFs
# --------------------------------------------------------------------------
def eof(X, w, rows=None):
    """Weighted EOFs of X[rows]. Returns dict with
    mu (N,), E (K x N) EOFs in weighted space, s (K,) singular values,
    pcs (Tr x K), frac (K,) variance fraction."""
    X = _check_X(X)
    w = safe_weights(w)
    if w.shape != (X.shape[1],):
        raise ValueError(f"w shape {w.shape} != (N,) = ({X.shape[1]},)")
    Xr = X if rows is None else X[rows[0]:rows[1]]
    if Xr.shape[0] < 2:
        raise ValueError(f"need >= 2 rows for EOFs, got {Xr.shape[0]}")
    mu = Xr.mean(axis=0)
    Z = (Xr - mu) * w
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    frac = s**2 / np.sum(s**2)
    return dict(mu=mu, E=Vt, s=s, pcs=U * s, frac=frac, w=w)


# --------------------------------------------------------------------------
# proxy locations
# --------------------------------------------------------------------------
def load_proxy_sites(path):
    """Return (lat, lon, archive, ids) from either
      * cfr's pages2kv2.json (pandas column-oriented JSON with keys
        geo_meanLat, geo_meanLon, archiveType, paleoData_pages2kID), or
      * a CSV with columns geo_meanLat, geo_meanLon [, archiveType, id]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"proxy site file not found: {path}")
    need = ("geo_meanLat", "geo_meanLon")
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        missing = [k for k in need if k not in d]
        if missing:
            raise KeyError(f"{path} lacks {missing}; keys: {list(d)[:20]}")
        keys = sorted(d["geo_meanLat"], key=int)
        lat = np.array([d["geo_meanLat"][k] for k in keys], float)
        lon = np.array([d["geo_meanLon"][k] for k in keys], float)
        arch = [d.get("archiveType", {}).get(k, "?") for k in keys]
        ids = [d.get("paleoData_pages2kID", {}).get(k, k) for k in keys]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{path} is empty")
        missing = [k for k in need if k not in rows[0]]
        if missing:
            raise KeyError(f"{path} lacks columns {missing}; has {list(rows[0])}")
        lat = np.array([float(r["geo_meanLat"]) for r in rows])
        lon = np.array([float(r["geo_meanLon"]) for r in rows])
        arch = [r.get("archiveType", "?") for r in rows]
        ids = [r.get("id", str(i)) for i, r in enumerate(rows)]
    bad = ~(np.isfinite(lat) & np.isfinite(lon))
    if bad.any():
        ask_continue(f"{bad.sum()} proxy records have no lat/lon: "
                     f"{[ids[i] for i in np.where(bad)[0]][:10]}")
        keep = ~bad
        lat, lon = lat[keep], lon[keep]
        arch = [a for a, k in zip(arch, keep) if k]
        ids = [i for i, k in zip(ids, keep) if k]
    return lat, lon, arch, ids


def _unit(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def nearest_cells(cell_lat, cell_lon, plat, plon, max_km=500.0):
    """Nearest grid cell (great-circle) for each proxy. Longitude convention
    (-180..180 vs 0..360) does not matter. Proxies farther than max_km from
    any valid cell are reported and skipped (ask quit/continue)."""
    R = 6371.0
    G = _unit(cell_lat, cell_lon)                 # (N, 3)
    Pp = _unit(plat, plon)                        # (M, 3)
    cosd = np.clip(Pp @ G.T, -1, 1)               # (M, N)
    idx = np.argmax(cosd, axis=1)
    dist = R * np.arccos(cosd[np.arange(len(idx)), idx])
    far = dist > max_km
    if far.any():
        ask_continue(f"{far.sum()} proxies are > {max_km} km from any valid cell "
                     f"(max {dist.max():.0f} km); they will be skipped")
    keep = np.where(~far)[0]
    return idx[keep], dist[keep], keep


def sample_pseudoproxies(X, idx, snr=0.5, rng=None):
    """P[:, i] = X[:, idx[i]] + noise, noise sd = std(signal_i) / snr.
    snr = inf gives perfect proxies. SNR is a ratio of standard deviations,
    the usual PPE convention (e.g. SNR 0.5 ~ proxy-truth corr 0.45)."""
    X = _check_X(X)
    rng = np.random.default_rng(rng)
    S = X[:, idx]
    if np.isinf(snr):
        return S.copy()
    if snr <= 0:
        raise ValueError(f"snr must be > 0, got {snr}")
    sd = S.std(axis=0) / snr
    return S + rng.standard_normal(S.shape) * sd


# --------------------------------------------------------------------------
# baseline reconstruction
# --------------------------------------------------------------------------
def reconstruct_eof(X, w, P, cal, r=4, lam=1e-3):
    """Inverse-regression EOF reconstruction (see module docstring).
    cal = (c0, c1) half-open row range for calibration. Truth X is used
    ONLY inside cal (that is the instrumental-era stand-in).
    lam is relative: actual ridge = lam * trace(A D A^T) / r."""
    X = _check_X(X)
    P = np.asarray(P, float)
    if P.shape[0] != X.shape[0]:
        raise ValueError(f"P has {P.shape[0]} rows, X has {X.shape[0]}")
    c0, c1 = cal
    Tc = c1 - c0
    if Tc <= r + 1:
        raise ValueError(f"calibration length {Tc} too short for r = {r}")
    ef = eof(X, w, rows=cal)
    if r > ef["E"].shape[0]:
        raise ValueError(f"r = {r} > available EOFs {ef['E'].shape[0]}")
    E, PCc = ef["E"][:r], ef["pcs"][:, :r]
    pmu = P[c0:c1].mean(axis=0)
    Pc = P[c0:c1] - pmu
    A, *_ = np.linalg.lstsq(PCc, Pc, rcond=None)          # (r, M)
    Rres = Pc - PCc @ A
    sig2 = Rres.var(axis=0, ddof=r)
    if np.any(sig2 <= 0):
        raise ValueError("zero calibration residual variance: proxies are exact "
                         "linear functions of the PCs; use snr < inf or lam > 0")
    D = 1.0 / sig2
    G = (A * D) @ A.T                                      # A D A^T (r x r)
    ridge = lam * np.trace(G) / r
    Bt = np.linalg.solve(G + ridge * np.eye(r), (A * D)).T  # (M, r)
    PChat = (P - pmu) @ Bt                                  # (T, r)
    Xhat = (PChat @ E) / ef["w"] + ef["mu"]
    return Xhat, dict(eof=ef, A=A, sig2=sig2, PChat=PChat, ridge=ridge)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(X, Xhat, cell_lat, val):
    v0, v1 = val
    x, y = X[v0:v1], Xhat[v0:v1]
    xa, ya = x - x.mean(0), y - y.mean(0)
    den = np.sqrt((xa**2).sum(0) * (ya**2).sum(0))
    corr = np.where(den > 0, (xa * ya).sum(0) / np.where(den > 0, den, 1), np.nan)
    ce = 1 - ((x - y) ** 2).sum(0) / (xa**2).sum(0)
    a = np.cos(np.deg2rad(cell_lat))
    a = a / a.sum()
    g, gh = x @ a, y @ a
    gmt_r = np.corrcoef(g, gh)[0, 1]
    gmt_ce = 1 - np.sum((g - gh) ** 2) / np.sum((g - g.mean()) ** 2)
    return dict(corr=corr, ce=ce, gmt=g, gmt_hat=gh, gmt_r=gmt_r, gmt_ce=gmt_ce,
                corr_area_mean=np.nansum(corr * a), ce_area_mean=np.sum(ce * a),
                corr_median=np.nanmedian(corr), ce_median=np.median(ce))


# --------------------------------------------------------------------------
# synthetic truth
# --------------------------------------------------------------------------
def synthetic_field(lat, lon, years, rng, noise_sd=0.35):
    """Four planted large-scale patterns x AR(1) series + white noise.
    Returns X (T x N), cell_lat, cell_lon (all cells kept), patterns (4 x N)."""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    cl, co = LAT.ravel(), LON.ravel()
    u = _unit(cl, co)

    def blob(la, lo, km):
        d = 6371 * np.arccos(np.clip(u @ _unit(np.array(la), np.array(lo)), -1, 1))
        return np.exp(-0.5 * (d / km) ** 2)

    pats = np.stack([
        0.6 + 0.8 * np.sin(np.deg2rad(cl)) ** 2,                      # global + polar amp.
        blob(0, 230, 2500) - 0.6 * blob(0, 140, 2000),                # ENSO-like
        blob(65, 330, 1500) - blob(38, 330, 1500),                    # NAO-like dipole
        np.where(cl < -40, np.cos(np.deg2rad(3 * (cl + 60))), 0.0),   # southern annular
    ])
    T = len(years)
    phis, amps = [0.9, 0.3, 0.2, 0.5], [0.25, 0.8, 0.7, 0.5]
    ts = np.zeros((T, 4))
    for k in range(4):
        e = rng.standard_normal(T) * np.sqrt(1 - phis[k] ** 2)
        for t in range(1, T):
            ts[t, k] = phis[k] * ts[t - 1, k] + e[t]
        ts[:, k] *= amps[k]
    # forcing on the global mode: random volcanic dips + post-1850 warming
    volc = np.zeros(T)
    for t in rng.choice(T, size=T // 40, replace=False):
        volc[t:t + 3] -= rng.uniform(0.3, 1.2) * np.array([1, 0.5, 0.2])[: T - t]
    ts[:, 0] += volc + np.clip(years - 1850, 0, None) / 150.0 * 1.0
    X = ts @ pats + noise_sd * rng.standard_normal((T, cl.size))
    return X, cl, co, pats


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def demo(pages2k=None, snr=0.5, r=4, seed=0, outdir=None):
    import matplotlib
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    lat = np.linspace(-90, 90, 46)                 # 4 deg, includes poles like CESM f19
    lon = np.arange(0, 360, 5.0)
    years = np.arange(850, 2006)                   # LME span [850, 2006)
    X, cl, co, _ = synthetic_field(lat, lon, years, rng)
    X = X - X[(years >= 1951) & (years < 1981)].mean(0)
    w = np.sqrt(np.clip(np.cos(np.deg2rad(cl)), 0, None))

    if pages2k:
        plat, plon, arch, ids = load_proxy_sites(pages2k)
        src = f"PAGES 2k sites from {os.path.basename(pages2k)}"
    else:
        M = 200
        plat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, M)))
        plon = rng.uniform(-180, 180, M)
        src = f"{M} random sites"
    idx, dist, keep = nearest_cells(cl, co, plat, plon)
    P = sample_pseudoproxies(X, idx, snr=snr, rng=rng)

    cal = tuple(np.searchsorted(years, [1850, 2006]))   # rows for [1850, 2006)
    val = tuple(np.searchsorted(years, [850, 1850]))    # rows for [850, 1850)
    Xhat, info = reconstruct_eof(X, w, P, cal, r=r)
    m = metrics(X, Xhat, cl, val)
    ef_full = eof(X, w)

    print(f"Synthetic truth: X {X.shape} (T x N), years [{years[0]}, {years[-1] + 1})")
    print(f"Proxies: {src}: {len(keep)} used, {np.unique(idx).size} distinct cells, "
          f"median snap distance {np.median(dist):.0f} km, SNR = {snr}")
    print(f"Calibration years [1850, 2006), validation [850, 1850), r = {r}")
    print(f"Variance explained, full-period EOFs k in [0, 6): "
          + ", ".join(f"{f:.3f}" for f in ef_full["frac"][:6]))
    print(f"GMT:   corr = {m['gmt_r']:.3f}   CE = {m['gmt_ce']:.3f}")
    print(f"Cells: corr area-mean = {m['corr_area_mean']:.3f}  median = "
          f"{m['corr_median']:.3f};  CE area-mean = {m['ce_area_mean']:.3f}  "
          f"median = {m['ce_median']:.3f}")
    print("r sweep (GMT corr / GMT CE / cell CE area-mean):")
    for rr in (1, 2, 4, 8, 16):
        Xh, _ = reconstruct_eof(X, w, P, cal, r=rr)
        mm = metrics(X, Xh, cl, val)
        print(f"  r = {rr:2d}:  {mm['gmt_r']:.3f} / {mm['gmt_ce']:.3f} / "
              f"{mm['ce_area_mean']:.3f}")

    # ---------------- figure ----------------
    yshow = int(years[val[0]:val[1]][np.argmin(m["gmt"])])   # coldest validation year
    t = int(np.searchsorted(years, yshow))
    shp = (lat.size, lon.size)
    vmax = np.nanpercentile(np.abs(X[t]), 98)
    ink, muted = "#222222", "#777777"
    fig = plt.figure(figsize=(12, 8.5), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    for k, (fld, ttl) in enumerate([(X[t], f"Truth, year {yshow}"),
                                    (Xhat[t], f"Reconstruction (r = {r}, SNR = {snr})")]):
        ax = fig.add_subplot(gs[0, k])
        pm = ax.pcolormesh(lon, lat, fld.reshape(shp), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="nearest")
        ax.scatter(np.mod(plon[keep], 360), plat[keep], s=6, c="k", lw=0,
                   label="proxy sites")
        ax.set_title(ttl, color=ink, fontsize=11)
        ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
        ax.set_xlabel("lon (deg E)", color=muted); ax.set_ylabel("lat", color=muted)
    fig.colorbar(pm, ax=fig.axes[:2], shrink=0.8, label="anomaly (synthetic units)")

    ax = fig.add_subplot(gs[1, 0])
    ce_map = m["ce"].reshape(shp)
    pm2 = ax.pcolormesh(lon, lat, np.clip(ce_map, -1, 1), cmap="RdBu_r", vmin=-1,
                        vmax=1, shading="nearest")
    ax.set_title("Validation CE per cell (clipped to [-1, 1])", color=ink, fontsize=11)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    fig.colorbar(pm2, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[1, 1])                # dot plot, not bars
    K = 20
    kk = np.arange(K)                             # zero-based mode index k
    fr = ef_full["frac"][:K]
    ax.plot(kk, fr, ls="none", marker="o", ms=8, mfc="white", mec="#2a6fdb", mew=2)
    ax.plot(kk[:r], fr[:r], ls="none", marker="o", ms=8, color="#2a6fdb")
    ax.set_yscale("log")
    ax.set_xticks(kk[::2])
    ax.set_xlabel("EOF index k (zero-based)", color=muted)
    ax.set_ylabel("fraction of variance", color=muted)
    ax.set_title(f"Variance explained (filled = retained, k in [0, {r}))", color=ink, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(gs[2, :])
    yv = years[val[0]:val[1]]
    ax.plot(yv, m["gmt"], lw=1.2, color=ink, label="truth")
    ax.plot(yv, m["gmt_hat"], lw=1.2, color="#d9480f", label="reconstruction")
    ax.set_title(f"Global mean, validation [850, 1850): corr {m['gmt_r']:.2f}, "
                 f"CE {m['gmt_ce']:.2f}", color=ink, fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel("year", color=muted)
    ax.grid(alpha=0.25, lw=0.5)

    outdir = outdir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{_dt.date.today():%Y-%m-%d}_eof_pseudoproxy_demo.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    plt.show()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages2k", help="cfr pages2kv2.json or CSV with geo_meanLat/geo_meanLon")
    ap.add_argument("--snr", type=float, default=0.5)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    demo(a.pages2k, a.snr, a.r, a.seed)
