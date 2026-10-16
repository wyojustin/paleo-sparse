"""
What did the learned interpolators learn? Interpretation only, not scoring.

Trains each model once on all sites (no site held out), using the same code
and budget as proxy_learned.py, then probes it:
  * effective weight of neighbor i: g_i = d zhat / d z_i(t), averaged over
    rows (for LSTM / TCN the total derivative: static input + sequence at tau = 0)
  * shrinkage: sum_i g_i (1 = copy a weighted mean, < 1 = shrink toward 0)
  * weight vs distance d_ij, binned
  * temporal kernel (LSTM, TCN): d zhat / d z_i(t + tau) summed over i, tau in [-H, H]
  * class pairs: mean g by (target class, neighbor class): learned sign flips?
  Kriging: the fitted correlogram rho(d) and its weights, for comparison.

Usage
  python src/proxy_probe.py --window 1500 2001
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy_learned as PL  # noqa: E402


def main(window, path, out_dir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"prep_{window[0]}_{window[1]}.npz")
    PL.prepare(window, path, npz)
    PL._init(npz)
    torch.set_num_threads(8)
    D = PL._D
    S = D["S"]
    obs = np.isfinite(S)
    tt, jj = np.nonzero(obs)
    nbr = PL.neighbors(-1, tt, jj)
    Xs = PL.static_features(tt, jj, nbr)
    y = S[tt, jj].astype(np.float32)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), PL.N_TRAIN, replace=False)
    Xt = PL.temporal_features(tt[sub], nbr[sub])
    probe = rng.choice(len(y), 6000, replace=False)
    Xs_p = torch.from_numpy(Xs[probe]).requires_grad_(True)
    Xt_p = torch.from_numpy(PL.temporal_features(tt[probe], nbr[probe])).requires_grad_(True)
    zcol = np.arange(PL.K) * 10                        # z_i position in each 10-wide neighbor block
    dist = D["Dkm"][jj[probe][:, None], nbr[probe]]    # (n, K)
    cls_t, cls_n = D["cls"][jj[probe]], D["cls"][nbr[probe]]

    res = {}
    A = np.c_[Xs, np.ones(len(Xs))].astype(np.float64)
    w = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ y)
    res["ridge"] = dict(g=np.broadcast_to(w[zcol], (len(probe), PL.K)), kern=None)
    for name, cls_ in PL.build_models(Xs.shape[1]).items():
        torch.manual_seed(0)
        m = PL._train(cls_(), (Xs[sub], Xt), y[sub], seed=0)
        Xs_p.grad = None; Xt_p.grad = None
        m(Xs_p, Xt_p).sum().backward()
        gs = Xs_p.grad.numpy()[:, zcol]                 # (n, K)
        gt = (Xt_p.grad.numpy()[:, :, :PL.K] if Xt_p.grad is not None   # (n, W, K) z channels
              else np.zeros((len(probe), 2 * PL.H + 1, PL.K), np.float32))   # MLP: no sequence input
        res[name] = dict(g=gs + gt[:, PL.H, :], kern=gt.sum(2).mean(0), gstatic=gs)
        print(f"  trained {name}")
    cg, f = PL.fit_correlogram(-1)
    print("\nKriging correlogram rho(d) = a exp(-d/L1) + b exp(-d/L2) + c:")
    for k, p in cg.items():
        print(f"  {k:5s}: a {p[0]:.2f}, L1 {p[1]:.0f} km, b {p[2]:.2f}, L2 {p[3]:.0f} km, c {p[4]:+.3f};"
              f"  rho(0+) = {p[0] + p[2] + p[4]:.2f}  -> nugget {1 - p[0] - p[2] - p[4]:.2f}")
    # kriging weights on the probe rows (target = j)
    t_p, j_p, n_p = tt[probe], jj[probe], nbr[probe]
    gk = np.zeros((len(probe), PL.K))
    for a in range(0, len(probe), 1000):
        sl = slice(a, a + 1000)
        Dn = D["Dkm"][n_p[sl][:, :, None], n_p[sl][:, None, :]]
        tr = D["tree"]
        tn = tr[n_p[sl]][:, :, None] & tr[n_p[sl]][:, None, :]
        rho = lambda X, T_: np.where(T_, f(X, *cg["tree"]), f(X, *cg["other"]))
        C = rho(Dn, tn); C[:, np.arange(PL.K), np.arange(PL.K)] = 1.0
        c = rho(D["Dkm"][j_p[sl][:, None], n_p[sl]], tr[j_p[sl]][:, None] & tr[n_p[sl]])
        gk[sl] = np.linalg.solve(C, c[..., None])[..., 0]
    res["krige"] = dict(g=gk, kern=None)

    names = ["ridge", "krige", "mlp", "lstm", "tcn"]
    print("\nEffective neighbor weights g_i = d zhat / d z_i (mean over probe rows)")
    print(f"  {'model':6s} sum g (shrinkage)   g by neighbor rank 0..{PL.K - 1}")
    for n_ in names:
        g = res[n_]["g"]
        print(f"  {n_:6s} {np.median(g.sum(1)):+.3f}             " +
              " ".join(f"{v:+.3f}" for v in g.mean(0)))
    print("\nTemporal kernel (sum over neighbors of d zhat / d z_i(t + tau)), tau in [-3, 3]:")
    for n_ in ("lstm", "tcn"):
        k = res[n_]["kern"]
        print(f"  {n_:5s} " + " ".join(f"{t:+d}:{k[PL.H + t]:+.3f}" for t in range(-3, 4)) +
              f"   |tau| > 3 total {np.abs(k).sum() - np.abs(k[PL.H - 3:PL.H + 4]).sum():.3f}")
    print("\nMean weight by (target class -> neighbor class), MLP | krige, pairs with >= 200 rows:")
    for a in range(5):
        for b in range(5):
            m_ = (cls_t[:, None] == a) & (cls_n == b)
            if m_.sum() >= 200:
                print(f"  {PL.CLASSES[a]:8s} <- {PL.CLASSES[b]:8s} n {m_.sum():6d}:  "
                      f"mlp {res['mlp']['g'][m_].mean():+.3f}  lstm {res['lstm']['g'][m_].mean():+.3f}  "
                      f"krige {res['krige']['g'][m_].mean():+.3f}")

    # ---------------- figure ----------------
    ink, muted = "#222222", "#777777"
    col = {"ridge": "#777777", "krige": "#222222", "mlp": "#2a6fdb", "lstm": "#d9480f", "tcn": "#2f9e44"}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    edges = np.geomspace(5, 5000, 14)
    xc = np.sqrt(edges[:-1] * edges[1:])
    for n_ in names:
        g = res[n_]["g"]
        yb = [np.mean(g[(dist >= a) & (dist < b)]) if np.any((dist >= a) & (dist < b)) else np.nan
              for a, b in zip(edges[:-1], edges[1:])]
        axs[0].plot(xc, yb, color=col[n_], marker="o", ms=4, lw=1.3, label=n_)
    axs[0].set_xscale("log"); axs[0].axhline(0, color=muted, lw=0.6)
    axs[0].set_xlabel("distance target-neighbor (km)", color=muted)
    axs[0].set_ylabel("mean weight d zhat / d z_i", color=muted)
    axs[0].set_title(f"Weight vs distance [{window[0]}, {window[1]})", color=ink, fontsize=10)
    axs[0].legend(frameon=False, fontsize=8); axs[0].grid(alpha=0.25, lw=0.5, which="both")

    for n_ in names:
        g = res[n_]["g"].sum(1)
        axs[1].plot(np.sort(g), np.linspace(0, 1, g.size), color=col[n_], lw=1.3, label=n_)
    axs[1].axvline(1, color=muted, lw=0.6, ls="--")
    axs[1].set_xlabel("sum of neighbor weights (1 = no shrinkage)", color=muted)
    axs[1].set_ylabel("fraction of rows (CDF)", color=muted)
    axs[1].set_title("Shrinkage", color=ink, fontsize=10)
    axs[1].set_xlim(-0.5, 1.5); axs[1].legend(frameon=False, fontsize=8); axs[1].grid(alpha=0.25, lw=0.5)

    tau = np.arange(-PL.H, PL.H + 1)
    for n_ in ("lstm", "tcn"):
        axs[2].plot(tau, res[n_]["kern"], color=col[n_], marker="o", ms=4, lw=1.3, label=n_)
    axs[2].axhline(0, color=muted, lw=0.6)
    axs[2].set_xlabel("lag tau (yr), neighbor year t + tau", color=muted)
    axs[2].set_ylabel("sum_i d zhat / d z_i(t + tau)", color=muted)
    axs[2].set_title("Learned temporal kernel", color=ink, fontsize=10)
    axs[2].legend(frameon=False, fontsize=8); axs[2].grid(alpha=0.25, lw=0.5)
    png = f"figures/{_dt.date.today().isoformat()}_proxy_probe_{window[0]}_{window[1]}.png"
    fig.savefig(png, dpi=130)
    print(f"\nwrote {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/cfr-data/pages2kv2.json")
    ap.add_argument("--window", type=int, nargs=2, default=[1500, 2001])
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    main(tuple(a.window), a.path, a.out, show=not a.no_show)
