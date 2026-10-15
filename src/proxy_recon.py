"""
Sparse (OMP / LASSO in a dictionary Psi) vs l2 reconstructions from the same
pseudoproxies at the PAGES 2k sites.

Conventions
  * Row-vector: X (T, N), proxies P (T, M), codes C (T, K), Psi (K, N).
  * Half-open ranges [a, b).
  * Never fail silently (same contract as the other scripts).

Model (pseudoproxies, so the forward operator is known)
      y_t = x_t Phi + eps_t,  Phi = cell selection (N x M),  eps ~ N(0, diag(sig2))
  sig2 is estimated per proxy from calibration residuals P_c - X_c Phi,
  years [1850, 2006): the same information the EOF regression baseline uses.
  All methods use the same "prior years" (rows pr) for their patterns:
      mu = mean_pr(X),  A = X_pr - mu (n x N)

  l2 methods (Gaussian prior on x):
    gauss      B = A^T A / (n - 1). Kalman / offline EnKF with the prior
               years as the ensemble (as in LMR, no localization):
                 xhat = mu + (y - mu Phi) S^{-1} (A Phi)^T A / (n - 1),
                 S = (A Phi)^T (A Phi) / (n - 1) + diag(sig2)
    gauss_r    same with A truncated to its leading r weighted EOFs
    eofreg_r   eof_pseudoproxy.reconstruct_eof (inverse regression, earlier result)

  l1 / sparse methods, Psi from the prior years (weighted space, as in
  sparsity_test.py, then unweighted: Psi = D V / w):
    whitened  yw = (y - mu Phi) / sig,  G = (Psi Phi) / sig, rows normalized
    omp_*     greedy, every L in [1, Lmax] (OMP is nested in L)
    omp_*_chi2  per year, the smallest L with ||yw - c G||^2 <= M
               (discrepancy principle: stop at the noise level; no tuning)
    lasso_*   FISTA on 1/2 ||yw - c G||^2 + lam ||c||_1, lam = alpha * median_t
               max|yw_t G^T|, swept over alpha
  Dictionaries: analog (normalized prior years) and K-SVD (K atoms, trained
  at L_train = 4).
  Picking the best L or alpha on validation years is an oracle choice and is
  labelled as such; chi2 OMP and the l2 methods are tuning-free.

Designs
  cal     prior = calibration [1850, 2006), test = [850, 1850). Same
          information as the EOF regression baseline.
  blocks  test = decade blocks with index % 5 == 2 inside [850, 1850),
          prior = every other year. Prior and truth come from one run
          (different years), the usual perfect-model PPE caveat.

Usage
  python src/proxy_recon.py data/*.TREFHT.*.nc --pages2k data/cfr-data/pages2kv2.json
  python src/proxy_recon.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eof_pseudoproxy import (load_proxy_sites, metrics, nearest_cells,  # noqa: E402
                             reconstruct_eof, safe_weights, sample_pseudoproxies)
from sparsity_test import block_split, ksvd, omp  # noqa: E402


# --------------------------------------------------------------------------
# l2: Gaussian prior, Kalman update
# --------------------------------------------------------------------------
def gauss_recon(A, mu, idx, sig2, P):
    """A (n, N) prior anomalies, mu (N,), idx (M,) cell of each proxy,
    sig2 (M,) noise variances, P (T, M). Returns Xhat (T, N)."""
    n = A.shape[0]
    if n < 2:
        raise ValueError(f"need >= 2 prior years, got {n}")
    AP = A[:, idx]                                     # (n, M)
    S = AP.T @ AP / (n - 1) + np.diag(sig2)
    Yc = P - mu[idx]
    Q = np.linalg.solve(S, Yc.T).T                     # (T, M) = Yc S^{-1}
    return mu + ((Q @ AP.T) / (n - 1)) @ A


# --------------------------------------------------------------------------
# l1: dictionary, whitened sensing matrix, OMP path and LASSO
# --------------------------------------------------------------------------
class Sensing:
    """Psi (K, N) field atoms. Builds G = normalized (Psi Phi)/sig."""

    def __init__(self, Psi, idx, sig2):
        sig = np.sqrt(sig2)
        G = Psi[:, idx] / sig                          # (K, M)
        self.scale = np.linalg.norm(G, axis=1)
        if np.any(self.scale < 1e-12):
            bad = np.where(self.scale < 1e-12)[0]
            raise ValueError(f"{bad.size} atoms are invisible to the proxies "
                             f"(zero at every proxy cell): {bad[:10]}")
        self.G = G / self.scale[:, None]
        self.Psi, self.idx, self.sig = Psi, idx, sig

    def whiten(self, P, mu):
        return (P - mu[self.idx]) / self.sig

    def field(self, Ct, mu):
        """Codes in normalized-G units -> field."""
        return mu + (Ct / self.scale) @ self.Psi


def lasso_path(Y, G, alphas, n_iter=400):
    """FISTA for each alpha (descending), warm started. Y (T, M), G (K, M)
    unit rows. lam = alpha * median_t max_k |Y_t G_k|. Returns list of C."""
    Lc = np.linalg.norm(G, 2) ** 2
    lmax = np.median(np.max(np.abs(Y @ G.T), axis=1))
    C = np.zeros((Y.shape[0], G.shape[0]))
    out = []
    for a in sorted(alphas, reverse=True):
        lam = a * lmax
        Z, t = C.copy(), 1.0
        for _ in range(n_iter):
            grad = (Z @ G - Y) @ G.T
            Cn = Z - grad / Lc
            Cn = np.sign(Cn) * np.maximum(np.abs(Cn) - lam / Lc, 0.0)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            Z = Cn + ((t - 1) / tn) * (Cn - C)
            C, t = Cn, tn
        out.append((a, C.copy()))
    return out


# --------------------------------------------------------------------------
# one design
# --------------------------------------------------------------------------
def run_design(X, w, cl, P, idx, pr, te, cal, Lmax, alphas, K, seed, log=print):
    """pr, te: boolean row masks. cal: (c0, c1) rows for sig2 and eofreg.
    Returns list of result dicts and a dict of Xhat for the maps."""
    if np.any(pr & te):
        raise ValueError("prior and test rows overlap")
    c0, c1 = cal
    if np.any(te[c0:c1]):
        raise ValueError("test rows overlap the calibration period")
    w = safe_weights(w)
    nte = int(te.sum())
    M = idx.size
    resid = P[c0:c1] - X[c0:c1][:, idx]
    sig2 = resid.var(axis=0, ddof=1)
    if np.any(sig2 <= 0):
        raise ValueError("zero calibration noise variance; use a finite SNR")

    mu = X[pr].mean(axis=0)
    A = X[pr] - mu
    out, keep = [], {}

    def score(name, Xh, **kw):
        m = metrics(X[te], Xh[te], cl, (0, nte))
        z = (X[te] - Xh[te]) * w
        z0 = (X[te] - X[te].mean(0)) * w
        r = dict(method=name, cell_ce=m["ce_area_mean"], gmt_r=m["gmt_r"],
                 gmt_ce=m["gmt_ce"], unexpl=np.sum(z**2) / np.sum(z0**2), **kw)
        out.append(r)
        return m

    t0 = time.time()
    Xh = gauss_recon(A, mu, idx, sig2, P)
    keep["gauss"] = score("gauss", Xh)["ce"]
    # truncated Gaussian: A_r from the leading r weighted EOFs of the prior years
    Zp = A * w
    U, s, Vt = np.linalg.svd(Zp, full_matrices=False)
    for r in (4, 16):
        Ar = ((U[:, :r] * s[:r]) @ Vt[:r]) / w
        score(f"gauss_{r}", gauss_recon(Ar, mu, idx, sig2, P), r=r)
    for r in (4, 16):
        Xh, _ = reconstruct_eof(X, w, P, cal, r=r)
        score(f"eofreg_{r}", Xh, r=r)
    log(f"    l2 methods: {time.time() - t0:.1f} s")

    keep_s = s > s[0] * 1e-10
    Vt = Vt[keep_s]
    Apr = Zp @ Vt.T                                    # prior in weighted coords
    dicts = {"analog": Apr / np.linalg.norm(Apr, axis=1, keepdims=True)}
    t0 = time.time()
    Dk, info = ksvd(Apr, K, 4, n_iter=30, rng=seed)
    dicts[f"ksvd{K}"] = Dk
    log(f"    K-SVD K={K} L_train=4: {time.time() - t0:.1f} s"
        + (f", {info['n_replaced']} atom reseeds" if info["n_replaced"] else ""))

    for dname, D in dicts.items():
        Psi = (D @ Vt) / w                             # (K, N) field atoms
        sen = Sensing(Psi, idx, sig2)
        Yw = sen.whiten(P, mu)
        t0 = time.time()
        Lm = min(Lmax, D.shape[0])
        Cs, rss = omp(Yw, sen.G, Lm, path=True)
        for L, Ct in enumerate(Cs, start=1):
            score(f"omp_{dname}", sen.field(Ct, mu), L=L)
        # discrepancy principle: smallest L with rss <= M, else Lm
        ok = rss <= M
        Lsel = np.where(ok.any(axis=1), ok.argmax(axis=1), Lm - 1)
        Cch = np.stack(Cs)[Lsel, np.arange(len(Lsel))]
        Lte = Lsel[te] + 1
        m = score(f"omp_{dname}_chi2", sen.field(Cch, mu),
                  L=float(Lte.mean()), note=f"L median {np.median(Lte):.0f}, "
                  f"[{Lte.min()}, {Lte.max()}]")
        keep[f"omp_{dname}_chi2"] = m["ce"]
        log(f"    OMP {dname}: {time.time() - t0:.1f} s, chi2 picks mean L "
            f"{Lte.mean():.1f} on test years")
        t0 = time.time()
        best = -np.inf
        for a, Ct in lasso_path(Yw, sen.G, alphas):
            nnz = (Ct[te] != 0).sum(axis=1)
            m = score(f"lasso_{dname}", sen.field(Ct, mu), alpha=a, L=float(nnz.mean()))
            if m["ce_area_mean"] > best:               # oracle alpha, for the maps only
                best, keep[f"lasso_{dname}"] = m["ce_area_mean"], m["ce"]
        log(f"    LASSO {dname}: {time.time() - t0:.1f} s")
    return out, keep


# --------------------------------------------------------------------------
# self-test: a field that IS sparse in a known dictionary
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(3)
    N, K0, L0, T, M = 400, 120, 3, 1500, 60
    D0 = rng.standard_normal((K0, N))
    D0 /= np.linalg.norm(D0, axis=1, keepdims=True)
    C0 = np.zeros((T, K0))
    for t in range(T):
        C0[t, rng.choice(K0, L0, replace=False)] = rng.choice([-1, 1], L0) * (1 + rng.random(L0))
    X = C0 @ D0
    w = np.ones(N)
    cl = np.zeros(N)
    idx = rng.choice(N, M, replace=False)
    P = sample_pseudoproxies(X, idx, snr=2.0, rng=rng)
    pr, te = np.arange(T) < 1000, np.arange(T) >= 1200
    Yc_mu = X[pr].mean(0)
    sig2 = (P[1000:1200] - X[1000:1200][:, idx]).var(0, ddof=1)
    g = gauss_recon(X[pr] - Yc_mu, Yc_mu, idx, sig2, P)
    sen = Sensing(D0, idx, sig2)
    Cs, rss = omp(sen.whiten(P, Yc_mu), sen.G, 6, path=True)
    o = sen.field(Cs[L0 - 1], Yc_mu)
    eg = metrics(X[te], g[te], cl, (0, te.sum()))["ce_median"]
    eo = metrics(X[te], o[te], cl, (0, te.sum()))["ce_median"]
    print(f"[selftest] planted L0={L0}, K0={K0}, M={M} < N={N}: median CE "
          f"gauss {eg:.3f}, OMP(true Psi) {eo:.3f}")
    assert eo > eg + 0.1, (eo, eg)
    # rss path is non-increasing and matches the explicit residual
    assert np.all(np.diff(rss, axis=1) <= 1e-8)
    Yw = sen.whiten(P, Yc_mu)
    assert np.allclose(rss[:, 2], np.sum((Yw - Cs[2] @ sen.G) ** 2, axis=1))
    # LASSO at small lam approaches least squares fit quality on the support
    (_, Cl), = lasso_path(Yw[:50], sen.G, [0.05])
    print(f"[selftest] LASSO alpha 0.05: mean nnz {(Cl != 0).sum(1).mean():.1f}")
    try:
        Sensing(np.zeros((2, N)), idx, sig2)
        raise AssertionError("expected an error")
    except ValueError as e:
        print(f"[selftest] correctly refused: {e}")
    print("[selftest] PASS")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(files, var, pages2k, snr, Lmax, seed, outdir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lme_loader import load_lme

    f = load_lme(files, var)
    X, years, cl, co = f.X, f.years, f.cell_lat, f.cell_lon
    for need in (850, 1849, 1850, 2005):
        if need not in years:
            raise SystemExit(f"proxy_recon: truth lacks year {need}; needs [850, 2006)")
    plat, plon, arch, ids = load_proxy_sites(pages2k)
    idx, dist, keepp = nearest_cells(cl, co, plat, plon)
    rng = np.random.default_rng(seed)
    P = sample_pseudoproxies(X, idx, snr=snr, rng=rng)
    cal = tuple(np.searchsorted(years, [1850, 2006]))
    print(f"CESM-LME {var}: X {X.shape}; {idx.size} PAGES 2k proxies in "
          f"{np.unique(idx).size} cells, SNR {snr}, seed {seed}")

    alphas = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    past = years < 1850
    btr, bte = block_split(years)
    designs = {
        "cal": (~past, past, 64, "prior = [1850, 2006), test = [850, 1850)"),
        "blocks": (~(bte & past), bte & past, 256,
                   "test = decade blocks (index % 5 == 2) in [850, 1850), prior = other years"),
    }
    allres, maps = [], {}
    for name, (pr, te, K, desc) in designs.items():
        print(f"\nDesign '{name}': {desc}  ({pr.sum()} prior, {te.sum()} test years)")
        res, keep = run_design(X, f.w, cl, P, idx, pr, te, cal, Lmax, alphas, K, seed)
        for r in res:
            r["design"] = name
        allres += res
        maps[name] = keep
        best = {}
        for r in res:
            m = r["method"]
            if m not in best or r["cell_ce"] > best[m]["cell_ce"]:
                best[m] = r
        print(f"  {'method':22s} cell CE   GMT r   GMT CE  unexpl   (best over L/alpha = oracle)")
        for m, r in best.items():
            tag = f"L={r['L']:.1f}" if "L" in r else ""
            tag += f" alpha={r['alpha']}" if "alpha" in r else ""
            print(f"  {m:22s} {r['cell_ce']:7.3f} {r['gmt_r']:7.3f} {r['gmt_ce']:7.3f} "
                  f"{r['unexpl']:7.3f}   {tag} {r.get('note', '')}")

    date = _dt.date.today().isoformat()
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{date}_proxy_recon_{var.lower()}_snr{snr:g}")
    keys = ["design", "method", "L", "alpha", "r", "cell_ce", "gmt_r", "gmt_ce", "unexpl", "note"]
    with open(stem + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in allres:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {stem}.csv")

    # ---------------- figure ----------------
    ink, muted = "#222222", "#777777"
    fig = plt.figure(figsize=(13, 10), layout="constrained")
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1, 0.9])
    l2 = {"gauss": ("#222222", "-"), "gauss_16": ("#222222", "--"),
          "gauss_4": ("#222222", ":"), "eofreg_16": ("#777777", "--"),
          "eofreg_4": ("#777777", ":")}
    col = {"analog": "#2f9e44", "ksvd": "#d9480f"}
    for j, name in enumerate(designs):
        rs = [r for r in allres if r["design"] == name]
        ax = sf_top.add_subplot(1, 2, j + 1)
        for m, (c, ls) in l2.items():
            v = [r["cell_ce"] for r in rs if r["method"] == m][0]
            ax.axhline(v, color=c, ls=ls, lw=1.1, label=f"{m} (l2)")
        for dname in sorted({r["method"].split("_")[1] for r in rs if r["method"].startswith("omp")}):
            c = col["analog" if dname == "analog" else "ksvd"]
            o = [r for r in rs if r["method"] == f"omp_{dname}"]
            ax.plot([r["L"] for r in o], [r["cell_ce"] for r in o], color=c, lw=1.4,
                    marker="o", ms=4, label=f"OMP, {dname}")
            la = [r for r in rs if r["method"] == f"lasso_{dname}"]
            ax.plot([max(r["L"], 1) for r in la], [r["cell_ce"] for r in la], color=c,
                    lw=1.0, ls="--", marker="s", ms=4, mfc="white", label=f"LASSO, {dname} (x = mean nnz)")
            ch = [r for r in rs if r["method"] == f"omp_{dname}_chi2"][0]
            ax.plot([ch["L"]], [ch["cell_ce"]], ls="none", marker="*", ms=14, color=c,
                    mec="k", mew=0.6, label=f"OMP {dname}, chi2 stop")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("atoms per year (L, or mean nonzeros for LASSO)", color=muted)
        ax.set_ylabel("validation cell CE (area mean)", color=muted)
        ax.set_title(f"'{name}': {designs[name][3]}", color=ink, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        if j == 1:
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    shp = f.mask.shape
    for j, name in enumerate(designs):
        kk = [k for k in maps[name] if k.startswith("lasso_ksvd")][0]
        for i, (lab, v) in enumerate([("gauss (l2), cell CE", maps[name]["gauss"]),
                                      (f"{kk} (oracle alpha) minus gauss, cell CE",
                                       maps[name][kk] - maps[name]["gauss"])]):
            ax = sf_bot.add_subplot(2, 2, 2 * j + i + 1)
            lim = 1 if i == 0 else 0.3
            pm = ax.pcolormesh(f.lon, f.lat, f.to_grid(np.clip(v, -lim, lim)).reshape(shp),
                               cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
            if i == 0:
                ax.scatter(np.mod(plon[keepp], 360), plat[keepp], s=3, c="k", lw=0)
            ax.set_title(f"'{name}': {lab}", color=ink, fontsize=9)
            ax.set_xlim(f.lon[0], f.lon[-1]); ax.set_ylim(-90, 90)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(pm, ax=ax, shrink=0.8)
    fig.suptitle(f"CESM-LME {var}, {idx.size} PAGES 2k pseudoproxies, SNR {snr}: "
                 f"sparse (l1) vs Gaussian (l2) priors", color=ink, fontsize=11)
    fig.savefig(stem + ".png", dpi=130)
    print(f"wrote {stem}.png")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--pages2k", default="data/cfr-data/pages2kv2.json")
    ap.add_argument("--snr", type=float, default=0.5)
    ap.add_argument("--Lmax", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.files:
        selftest()
    else:
        main(a.files, a.var, a.pages2k, a.snr, a.Lmax, a.seed, a.outdir, show=not a.no_show)
