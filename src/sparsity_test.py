"""
Premise test: is the LME field sparser in a learned dictionary than in EOFs?

No proxies. On held-out years, compare approximation error at equal
sparsity L (number of nonzero coefficients per year).

Conventions
  * Row-vector: Z is (T, N), one row per year; codes C are (T, K); a
    dictionary D is (K, N), one atom per row; Zhat = C @ D.
  * Half-open ranges [a, b).
  * Never fail silently (same contract as lme_loader / eof_pseudoproxy).

Math
  Train rows tr, test rows te. mu = mean_tr(X), w = sqrt(cos lat),
      Z = (X - mu) * w                      (area-weighted anomalies)
  Train SVD Z_tr = U S V^T. Every method below builds atoms from training
  data, so atoms lie in span(V); work in coordinates A = Z V^T (lossless
  for training rows). For test rows the part outside span(V),
      floor = ||Z_te||^2 - ||Z_te V^T||^2,
  is unreachable by every method and is added back to each error.
  Score: unexplained fraction  e = ||Z_te - Zhat_te||_F^2 / ||Z_te||_F^2.

  Methods at sparsity L:
    eof_top   keep EOFs k in [0, L)                    (linear, fixed subspace)
    eof_best  keep the L largest |A[t, k]| per year    (sparse, orthonormal basis)
    analog    atoms = normalized training years, OMP L (exemplar dictionary)
    ksvd_K    K atoms learned by approximate K-SVD at sparsity L, OMP L
  OMP is batch OMP on the Gram matrix G = D D^T, vectorized over years.

Caveat: an L-sparse code in K atoms also carries log2 C(K, L) bits of
support, which L leading-EOF coefficients do not. This test only asks
whether the premise (sparse in a learned D) holds at all.

Usage
  python src/sparsity_test.py data/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.*.nc
  python src/sparsity_test.py --selftest
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
from eof_pseudoproxy import ask_continue, safe_weights  # noqa: E402


# --------------------------------------------------------------------------
# batch OMP
# --------------------------------------------------------------------------
def omp(A, D, L, G=None):
    """Batch OMP, all rows at once. A (n, R) signals, D (K, R) atoms with
    unit norm. Returns dense codes C (n, K) with exactly min(L, K) nonzeros
    per row (fewer only if a row is fully explained)."""
    n, K = A.shape[0], D.shape[0]
    if L > K:
        raise ValueError(f"OMP sparsity L = {L} > number of atoms K = {K}")
    nrm = np.linalg.norm(D, axis=1)
    if np.any(np.abs(nrm - 1) > 1e-6):
        raise ValueError("OMP needs unit-norm atoms")
    G = D @ D.T if G is None else G
    a0 = A @ D.T                                   # (n, K) correlations
    alpha = a0.copy()
    S = np.zeros((n, L), dtype=np.int64)
    rows = np.arange(n)
    c = np.zeros((n, 0))
    for s in range(L):
        score = np.abs(alpha)
        if s:
            score[rows[:, None], S[:, :s]] = -np.inf  # never reselect
        S[:, s] = np.argmax(score, axis=1)
        idx = S[:, :s + 1]
        Gs = G[idx[:, :, None], idx[:, None, :]]   # (n, s+1, s+1)
        Gs += 1e-10 * np.eye(s + 1)                # guards duplicate atoms
        c = np.linalg.solve(Gs, a0[rows[:, None], idx][..., None])[..., 0]
        alpha = a0 - np.einsum("ns,nsk->nk", c, G[idx])
    C = np.zeros((n, K))
    C[rows[:, None], S] = c
    return C


# --------------------------------------------------------------------------
# approximate K-SVD (Rubinstein, Zibulevsky & Elad 2008), row convention
# --------------------------------------------------------------------------
def ksvd(A, K, L, n_iter=30, rng=None, verbose=False):
    """Learn D (K, R) so that A ~ C D with L-sparse rows of C.
    Init: K distinct normalized training rows. Unused atoms are replaced by
    the worst-represented training row (counted and reported)."""
    rng = np.random.default_rng(rng)
    n, R = A.shape
    if K > n:
        raise ValueError(f"K = {K} atoms > {n} training rows; cannot initialize")
    D = A[rng.choice(n, K, replace=False)].copy()
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    n_replaced = 0
    hist = []
    for it in range(n_iter):
        C = omp(A, D, L)
        E = A - C @ D                               # residual (n, R)
        for k in rng.permutation(K):
            I = np.nonzero(C[:, k])[0]
            if I.size == 0:                         # dead atom: reseed
                worst = np.argmax(np.einsum("nr,nr->n", E, E))
                d = E[worst] / np.linalg.norm(E[worst])
                D[k] = d
                n_replaced += 1
                continue
            g = C[I, k]
            EI = E[I] + np.outer(g, D[k])           # residual without atom k
            d = g @ EI
            d /= np.linalg.norm(d)
            g = EI @ d
            D[k] = d
            C[I, k] = g
            E[I] = EI - np.outer(g, d)
        err = np.sum(E**2) / np.sum(A**2)
        hist.append(err)
        if verbose:
            print(f"    ksvd K={K} L={L} iter {it}: train unexplained {err:.4f}")
    return D, dict(hist=np.array(hist), n_replaced=n_replaced)


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------
def unexplained(A_true, A_hat, floor, tot):
    return (np.sum((A_true - A_hat) ** 2) + floor) / tot


def compare(X, w, tr, te, Ls, Ks, n_iter=30, seed=0, log=print):
    """tr, te: boolean row masks. Returns (rows of results, dict of extras)."""
    if np.any(tr & te):
        raise ValueError("train and test rows overlap")
    if tr.sum() < 2 or te.sum() < 1:
        raise ValueError(f"too few rows: train {tr.sum()}, test {te.sum()}")
    w = safe_weights(w)
    mu = X[tr].mean(axis=0)
    Ztr, Zte = (X[tr] - mu) * w, (X[te] - mu) * w
    U, s, Vt = np.linalg.svd(Ztr, full_matrices=False)
    keep = s > s[0] * 1e-10
    Vt = Vt[keep]
    Atr, Ate = Ztr @ Vt.T, Zte @ Vt.T
    tot_te, tot_tr = np.sum(Zte**2), np.sum(Ztr**2)
    floor_te = tot_te - np.sum(Ate**2)
    floor_tr = max(tot_tr - np.sum(Atr**2), 0.0)
    log(f"  train {tr.sum()} yrs, test {te.sum()} yrs, span dim R = {Vt.shape[0]}, "
        f"test out-of-span floor = {floor_te / tot_te:.4f}")

    out = []

    def add(method, K, L, e_te, e_tr=np.nan, extra=""):
        out.append(dict(method=method, K=K, L=L, test=e_te, train=e_tr, note=extra))

    analog = Atr / np.linalg.norm(Atr, axis=1, keepdims=True)
    Ga = analog @ analog.T
    extras = {"floor": floor_te / tot_te, "dicts": {}, "Vt": Vt}
    for L in Ls:
        # EOF, top L
        Ah = np.zeros_like(Ate); Ah[:, :L] = Ate[:, :L]
        Ahr = np.zeros_like(Atr); Ahr[:, :L] = Atr[:, :L]
        add("eof_top", Vt.shape[0], L, unexplained(Ate, Ah, floor_te, tot_te),
            unexplained(Atr, Ahr, floor_tr, tot_tr))
        # EOF, best L per year
        def best(Am):
            o = np.argsort(-np.abs(Am), axis=1)[:, :L]
            B = np.zeros_like(Am)
            r = np.arange(Am.shape[0])[:, None]
            B[r, o] = Am[r, o]
            return B
        add("eof_best", Vt.shape[0], L, unexplained(Ate, best(Ate), floor_te, tot_te),
            unexplained(Atr, best(Atr), floor_tr, tot_tr))
        # analog dictionary: training years themselves (train error is trivially
        # 0 at L >= 1, so it is not reported)
        Ca = omp(Ate, analog, L, G=Ga)
        add("analog", analog.shape[0], L, unexplained(Ate, Ca @ analog, floor_te, tot_te))
        # learned
        for K in Ks:
            if L > K:
                continue
            t0 = time.time()
            D, info = ksvd(Atr, K, L, n_iter=n_iter, rng=seed)
            Cte = omp(Ate, D, L)
            Ctr = omp(Atr, D, L)
            e_te = unexplained(Ate, Cte @ D, floor_te, tot_te)
            e_tr = unexplained(Atr, Ctr @ D, floor_tr, tot_tr)
            note = f"{info['n_replaced']} atom reseeds" if info["n_replaced"] else ""
            add(f"ksvd_{K}", K, L, e_te, e_tr, note)
            extras["dicts"][(K, L)] = (D, Ctr)
            log(f"    ksvd K={K:4d} L={L:3d}: test {e_te:.4f}  train {e_tr:.4f}  "
                f"({time.time() - t0:.1f} s{', ' + note if note else ''})")
        r = {m["method"]: m["test"] for m in out if m["L"] == L}
        log(f"  L={L:3d}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
    return out, extras


def block_split(years, block=10, period=5, phase=2):
    """Test = decade blocks whose index % period == phase (20 % of years).
    Blocks, not single years, so autocorrelation does not leak."""
    b = (years - years[0]) // block
    te = (b % period) == phase
    return ~te, te


# --------------------------------------------------------------------------
# self-test on a field that IS sparse in a known overcomplete dictionary
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    R, K0, L0, n = 40, 80, 3, 1200
    D0 = rng.standard_normal((K0, R))
    D0 /= np.linalg.norm(D0, axis=1, keepdims=True)
    C0 = np.zeros((n, K0))
    for i in range(n):
        C0[i, rng.choice(K0, L0, replace=False)] = rng.standard_normal(L0) + 2 * rng.choice([-1, 1], L0)
    A = C0 @ D0 + 0.01 * rng.standard_normal((n, R))
    # OMP recovers support with the true dictionary
    C = omp(A[:200], D0, L0)
    hit = np.mean([(set(np.nonzero(C[i])[0]) == set(np.nonzero(C0[i])[0])) for i in range(200)])
    print(f"[selftest] OMP exact-support rate with true D: {hit:.3f}")
    assert hit > 0.95, hit
    # K-SVD learns it: held-out error at L0 well below EOF-top-L0
    X = A
    years = np.arange(n)
    tr, te = years < 1000, years >= 1000
    res, _ = compare(X, np.ones(R), tr, te, Ls=[L0], Ks=[K0], n_iter=40, seed=0,
                     log=lambda *a: None)
    r = {m["method"]: m["test"] for m in res}
    print(f"[selftest] L={L0}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
    assert r[f"ksvd_{K0}"] < 0.1 < r["eof_top"], r
    # loud failures
    for bad in (lambda: omp(A, D0 * 2, 2), lambda: omp(A, D0, K0 + 1),
                lambda: compare(X, np.ones(R), tr, tr, [1], [4])):
        try:
            bad()
            raise AssertionError("expected an error")
        except ValueError as e:
            print(f"[selftest] correctly refused: {e}")
    print("[selftest] PASS")


# --------------------------------------------------------------------------
# main: real LME
# --------------------------------------------------------------------------
def run(files, var, Ls, Ks, n_iter, seed, outdir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lme_loader import load_lme

    f = load_lme(files, var)
    X, years = f.X, f.years
    for need in (850, 2005):
        if need not in years:
            raise SystemExit(f"sparsity_test: truth lacks year {need}; needs [850, 2006)")
    print(f"CESM-LME {var}: X {X.shape} (T x N), years [{years[0]}, {years[-1] + 1})")

    splits = {}
    tr, te = block_split(years)
    splits["blocks"] = (tr, te, "test = decades with index % 5 == 2 (20 %), train = rest")
    splits["future"] = (years < 1850, years >= 1850,
                        "train [850, 1850), test [1850, 2006) (forced warming)")
    allres, extras = [], {}
    for name, (tr, te, desc) in splits.items():
        print(f"\nSplit '{name}': {desc}")
        res, ex = compare(X, f.w, tr, te, Ls, Ks, n_iter=n_iter, seed=seed)
        for r in res:
            r["split"] = name
        allres += res
        extras[name] = ex

    date = _dt.date.today().isoformat()
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{date}_sparsity_premise_{var.lower()}.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["split", "method", "K", "L", "test", "train", "note"])
        wr.writeheader()
        wr.writerows(allres)
    print(f"\nwrote {csv_path}")

    # ---------------- figure ----------------
    ink, muted = "#222222", "#777777"
    style = {"eof_top": ("#222222", "o", "EOF, top L"),
             "eof_best": ("#777777", "s", "EOF, best L per year"),
             "analog": ("#2f9e44", "^", "analog (training years), OMP")}
    kcol = ["#2a6fdb", "#d9480f", "#7048e8", "#c2255c"]
    for i, K in enumerate(Ks):
        style[f"ksvd_{K}"] = (kcol[i % len(kcol)], "D", f"K-SVD, K = {K}, OMP")
    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.1, 1])
    for j, name in enumerate(splits):
        ax = sf_top.add_subplot(1, 2, j + 1)
        for meth, (col, mk, lab) in style.items():
            rr = [r for r in allres if r["split"] == name and r["method"] == meth]
            if not rr:
                continue
            Lv = [r["L"] for r in rr]
            ax.plot(Lv, [r["test"] for r in rr], color=col, marker=mk, ms=6, lw=1.4, label=lab)
            if meth.startswith("ksvd"):
                ax.plot(Lv, [r["train"] for r in rr], color=col, ls=":", lw=1.0)
        ax.axhline(extras[name]["floor"], color=muted, ls="--", lw=0.8)
        ax.text(Ls[0], extras[name]["floor"], "  out-of-span floor", color=muted,
                va="bottom", fontsize=8)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
        ax.set_xlabel("sparsity L (nonzeros per year)", color=muted)
        ax.set_ylabel("held-out unexplained fraction", color=muted)
        ax.set_title(f"{var}, split '{name}': {splits[name][2]}", color=ink, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        if j == 0:
            ax.legend(frameon=False, fontsize=8, title="dotted = K-SVD train error",
                      title_fontsize=8)

    # most-used atoms of the largest dictionary at a middle L (blocks split)
    Kb = max(Ks)
    Lb = [L for L in Ls if L <= 8 and (Kb, L) in extras["blocks"]["dicts"]][-1]
    D, Ctr = extras["blocks"]["dicts"][(Kb, Lb)]
    Vt = extras["blocks"]["Vt"]
    use = (Ctr != 0).sum(axis=0)
    order = np.argsort(-use)[:6]
    w = safe_weights(f.w)
    shp = f.mask.shape
    for i, k in enumerate(order):
        ax = sf_bot.add_subplot(2, 3, i + 1)
        atom = (D[k] @ Vt) / w
        v = np.nanpercentile(np.abs(atom), 99)
        ax.pcolormesh(f.lon, f.lat, f.to_grid(atom).reshape(shp), cmap="RdBu_r",
                      vmin=-v, vmax=v, shading="nearest")
        ax.set_title(f"atom {k}: used in {use[k]} of {Ctr.shape[0]} train yrs",
                     color=ink, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    sf_bot.suptitle(f"Most-used atoms, K-SVD K = {Kb}, L = {Lb}, split 'blocks' "
                    f"(sign arbitrary, unweighted units)", color=muted, fontsize=10)
    png = os.path.join(outdir, f"{date}_sparsity_premise_{var.lower()}.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--L", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--K", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.files:
        selftest()
    else:
        run(a.files, a.var, sorted(a.L), sorted(a.K), a.iters, a.seed, a.outdir,
            show=not a.no_show)
