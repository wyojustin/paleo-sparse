"""
Learned interpolators vs no-learning baselines, leave-one-site-out, on
PAGES 2k annual proxies. Measurement-only track: no model, no reanalysis.

Learning problem (no gridded truth exists)
  Learn f: (k nearest observed neighbors of a site, site metadata) -> z.
  For held-out site s, training pairs are every other site j != s as a
  target, in every year it is observed, with neighbors drawn from sites
  other than s and j. Site s never appears in training, as target or input.
  Then predict s from its own k nearest observed neighbors.

Inputs, target j in year t, neighbors i in [0, K) ordered by distance
  per neighbor  z_i(t), d_ij / 1000 km, (east, north)_ij / 1000 km in j's
                tangent plane, same-class flag, class one-hot (5)
  per target    class one-hot (5), |lat_j| / 90
  temporal      z_i(t + tau), mask_i(t + tau) for tau in [-H, H]
  classes       0 TRW, 1 density (MXD), 2 isotope (d18O, dD), 3 Sr/Ca, 4 other
  The class features let a model learn sign flips between proxy types.

Models
  ridge     linear in the static features (control for the MLP)
  krige     simple kriging (Gaussian process) with the empirical correlogram
            rho(d) = a exp(-d / L1) + b exp(-d / L2) + c, fitted to binned
            pair correlations not involving s, separately for tree-tree and
            other pairs; diag 1 (so the nugget 1 - rho(0+) is noise):
              zhat = rho_s^T C^{-1} z_nbrs
  mlp       static features -> 64 -> 64 -> 1
  lstm      bidirectional LSTM (hidden 32) over the neighbor sequence, center
            output + static embedding -> 1
  tcn       1-D CNN over time (kernel 5, dilations 1, 2, 4), center + static -> 1
  Neural nets: Adam, MSE, fixed budget (same for all), seeded per fold.

Scores per site (see proxy_baselines.py): r, v = var(zhat) / var(z), SS.

Usage
  python src/proxy_learned.py --window 1500 2001 [--sites 0 10] [--workers 16]
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K = 8          # neighbors
H = 10         # temporal half-width, sequence length 2H + 1
N_TRAIN = 60000
EPOCHS = 4
CLASSES = ["TRW", "density", "isotope", "Sr/Ca", "other"]


def proxy_class(p):
    if "Sr/Ca" in p:                                  # 'Sr/Ca' contains the separator
        parts = ["Sr/Ca"] + [q for q in p.replace("Sr/Ca", "").split("/") if q]
    else:
        parts = p.split("/")

    def one(q):
        if q == "TRW":
            return 0
        if q in ("MXD", "delta Density", "density"):
            return 1
        if q in ("d18O", "dD", "foram d18O"):
            return 2
        if q in ("Sr/Ca", "Coral Sr/Ca"):
            return 3
        return 4
    c = {one(q) for q in parts}
    if c == {2, 3}:                                   # coral Sr/Ca + d18O: both SST-negative
        return 3
    return c.pop() if len(c) == 1 else 4


# --------------------------------------------------------------------------
# shared preprocessing (main process), saved to an npz the workers load
# --------------------------------------------------------------------------
def prepare(window, path, out):
    from proxy_data import load_pages2k
    from proxy_baselines import site_matrix, unit
    recs, _ = load_pages2k(path)
    S, years, sites, rep = site_matrix(recs, window, log=lambda *a: None)
    lat = np.array([s["lat"] for s in sites]); lon = np.array([s["lon"] for s in sites])
    U = unit(lat, lon)
    Dkm = 6371.0 * np.arccos(np.clip(U @ U.T, -1, 1))
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    E = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    Nn = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    EN = 6371.0 * np.stack([U @ E.T, U @ Nn.T], -1).transpose(1, 0, 2)   # EN[j, i] offsets of i from j
    cls = np.array([proxy_class(s["proxy"]) for s in sites])
    tree = np.array([s["archive"] == "tree" for s in sites])
    obs = np.isfinite(S)
    T, Ns = S.shape
    nb = np.zeros((T, Ns, K + 1), np.int64)
    for t in range(T):
        Dt = np.where(obs[t][None, :], Dkm, np.inf)
        np.fill_diagonal(Dt, np.inf)
        part = np.argpartition(Dt, K + 1, axis=1)[:, :K + 1]
        o = np.argsort(np.take_along_axis(Dt, part, 1), axis=1)
        nb[t] = np.take_along_axis(part, o, 1)
        if np.any(~np.isfinite(np.take_along_axis(Dt, nb[t], 1))):
            raise RuntimeError(f"year {years[t]}: fewer than {K + 1} observed sites")
    # pairwise correlations over shared years (>= 50) for the correlogram
    F = obs.astype(float); S0 = np.nan_to_num(S)
    n = F.T @ F; sx = S0.T @ F; sxx = (S0 ** 2).T @ F; sxy = S0.T @ S0
    with np.errstate(all="ignore"):
        mx, my = sx / n, sx.T / n
        R = (sxy / n - mx * my) / np.sqrt((sxx / n - mx ** 2) * (sxx.T / n - my ** 2))
    R[n < 50] = np.nan
    np.savez(out, S=S, years=years, U=U, Dkm=Dkm, EN=EN, cls=cls, tree=tree, nb=nb, R=R,
             lat=lat, archive=np.array([s["archive"] for s in sites]),
             proxy=np.array([s["proxy"] for s in sites]), ids=np.array([s["ids"] for s in sites]))
    return rep, S.shape


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------
_D = {}


def _init(npz):
    import torch
    torch.set_num_threads(1)
    d = np.load(npz)
    _D.update({k: d[k] for k in d.files})


def neighbors(s, t, j):
    """K nearest observed neighbors of targets j in years t, excluding site s."""
    cand = _D["nb"][t, j]                             # (n, K + 1)
    o = np.argsort(cand == s, axis=1, kind="stable")
    return np.take_along_axis(cand, o, 1)[:, :K]


def static_features(t, j, nbr):
    S, Dkm, EN, cls, lat = _D["S"], _D["Dkm"], _D["EN"], _D["cls"], _D["lat"]
    z = S[t[:, None], nbr]
    d = Dkm[j[:, None], nbr] / 1000.0
    en = EN[j[:, None], nbr] / 1000.0
    same = (cls[nbr] == cls[j][:, None]).astype(float)
    oh = np.eye(5)[cls[nbr]]                          # (n, K, 5)
    per = np.concatenate([z[..., None], d[..., None], en, same[..., None], oh], -1)
    tgt = np.concatenate([np.eye(5)[cls[j]], (np.abs(lat[j]) / 90.0)[:, None]], -1)
    return np.concatenate([per.reshape(len(t), -1), tgt], -1).astype(np.float32)


def temporal_features(t, nbr):
    S = _D["S"]
    T = S.shape[0]
    tau = np.arange(-H, H + 1)
    tt = t[:, None] + tau[None, :]                    # (n, W)
    inside = (tt >= 0) & (tt < T)
    tt = np.clip(tt, 0, T - 1)
    z = S[tt[:, :, None], nbr[:, None, :]]            # (n, W, K)
    m = np.isfinite(z) & inside[..., None]
    z = np.where(m, z, 0.0)
    return np.concatenate([z, m.astype(float)], -1).astype(np.float32)


def fit_correlogram(s):
    from scipy.optimize import curve_fit
    R, Dkm, tree = _D["R"], _D["Dkm"], _D["tree"]
    Ns = R.shape[0]
    iu = np.triu_indices(Ns, 1)
    ok = np.isfinite(R[iu]) & (iu[0] != s) & (iu[1] != s)
    d, r = Dkm[iu][ok], R[iu][ok]
    tt = (tree[iu[0]] & tree[iu[1]])[ok]
    edges = np.r_[0, np.geomspace(25, 20100, 24)]
    f = lambda x, a, L1, b, L2, c: a * np.exp(-x / L1) + b * np.exp(-x / L2) + c
    out = {}
    for name, sel in (("tree", tt), ("other", ~tt)):
        dd, rr = d[sel], r[sel]
        xb, yb, wb = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (dd >= a) & (dd < b)
            if m.sum() >= 5:
                xb.append(np.median(dd[m])); yb.append(np.median(rr[m])); wb.append(m.sum())
        xb, yb, wb = map(np.array, (xb, yb, wb))
        p, _ = curve_fit(f, xb, yb, p0=(0.2, 50, 0.2, 700, 0.02), sigma=1 / np.sqrt(wb),
                         bounds=([0, 1, 0, 100, -0.1], [1, 300, 1, 5000, 0.2]), maxfev=20000)
        if p[0] + p[2] + p[4] >= 1:
            raise RuntimeError(f"correlogram fit for {name} pairs exceeds 1 at d = 0: {p}")
        out[name] = p
    return out, f


def krige(s, t, nbr, cg):
    p, f = cg
    Dkm, tree, S = _D["Dkm"], _D["tree"], _D["S"]
    rho = lambda D, tt: np.where(tt, f(D, *p["tree"]), f(D, *p["other"]))
    Dn = Dkm[nbr[:, :, None], nbr[:, None, :]]                     # (n, K, K)
    tn = tree[nbr][:, :, None] & tree[nbr][:, None, :]
    C = rho(Dn, tn)
    C[:, np.arange(K), np.arange(K)] = 1.0
    c = rho(Dkm[s, nbr], tree[s] & tree[nbr])                      # (n, K)
    w = np.linalg.solve(C, c[..., None])[..., 0]
    return np.sum(w * S[t[:, None], nbr], 1), w


def _train(model, inputs, y, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = y.shape[0]
    yt = torch.from_numpy(y)
    xs = [torch.from_numpy(x) for x in inputs]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            opt.zero_grad()
            loss = ((model(*[x[idx] for x in xs]).squeeze(-1) - yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    return model


def build_models(n_static):
    import torch
    nn = torch.nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(n_static, 64), nn.GELU(), nn.Linear(64, 64),
                                   nn.GELU(), nn.Linear(64, 1))

        def forward(self, xs, xt):
            return self.f(xs)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = nn.LSTM(2 * K, 32, batch_first=True, bidirectional=True)
            self.e = nn.Sequential(nn.Linear(n_static, 32), nn.GELU())
            self.h = nn.Sequential(nn.Linear(64 + 32, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, xs, xt):
            o = self.r(xt)[0][:, H]
            return self.h(torch.cat([o, self.e(xs)], -1))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(2 * K, 32, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
                                   nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.GELU())
            self.e = nn.Sequential(nn.Linear(n_static, 32), nn.GELU())
            self.h = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, xs, xt):
            o = self.c(xt.transpose(1, 2))[:, :, H]
            return self.h(torch.cat([o, self.e(xs)], -1))

    return {"mlp": MLP, "lstm": LSTM, "tcn": TCN}


def run_fold(s):
    import torch
    S = _D["S"]
    T, Ns = S.shape
    obs = np.isfinite(S)
    rng = np.random.default_rng(s)
    tt, jj = np.nonzero(obs)
    keep = jj != s
    tt, jj = tt[keep], jj[keep]
    nbr = neighbors(s, tt, jj)
    Xs = static_features(tt, jj, nbr)
    y = S[tt, jj].astype(np.float32)
    # test
    t_te = np.nonzero(obs[:, s])[0]
    nbr_te = _D["nb"][t_te, s][:, :K]
    Xs_te = static_features(t_te, np.full(t_te.size, s), nbr_te)
    preds = {}
    # ridge on all training rows
    A = np.c_[Xs, np.ones(len(Xs))].astype(np.float64)
    w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)
    preds["ridge"] = np.c_[Xs_te, np.ones(len(Xs_te))] @ w
    # kriging
    cg = fit_correlogram(s)
    preds["krige"], wk = krige(s, t_te, nbr_te, cg)
    # neural nets, same subsample
    sub = rng.choice(len(y), size=min(N_TRAIN, len(y)), replace=False)
    Xt = temporal_features(tt[sub], nbr[sub])
    Xt_te = temporal_features(t_te, nbr_te)
    for name, cls_ in build_models(Xs.shape[1]).items():
        torch.manual_seed(s)
        m = _train(cls_(), (Xs[sub], Xt), y[sub], seed=s)
        with torch.no_grad():
            preds[name] = m(torch.from_numpy(Xs_te), torch.from_numpy(Xt_te)).squeeze(-1).numpy()
    full = {k: np.full(T, np.nan) for k in preds}
    for k, v in preds.items():
        full[k][t_te] = v
    return s, full, dict(krige_wsum=float(np.median(wk.sum(1))), cg=cg[0])


# --------------------------------------------------------------------------
def main(window, path, sites_range, workers, out_dir):
    from proxy_baselines import predict_delaunay, predict_nearest, score
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{window[0]}_{window[1]}"
    npz = os.path.join(out_dir, f"prep_{tag}.npz")
    rep, shape = prepare(window, path, npz)
    print(f"Window [{window[0]}, {window[1]}): S {shape} (T x Ns); " +
          "; ".join(f"{k} {v}" for k, v in rep.items()))
    d = np.load(npz)
    S, U = d["S"], d["U"]
    Ns = S.shape[1]
    todo = list(range(*sites_range)) if sites_range else list(range(Ns))
    t0 = time.time()
    preds = {}
    info = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(npz,)) as pool:
        for k, (s, full, inf) in enumerate(pool.imap_unordered(run_fold, todo)):
            preds[s], info[s] = full, inf
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print(f"  folds {k + 1}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
    # baselines on the same sites
    cache = {}
    for s in todo:
        preds[s]["nearest"] = predict_nearest(S, U, s)[0]
        preds[s]["delaunay"] = predict_delaunay(S, U, s, cache)[0]
    names = ["nearest", "delaunay", "ridge", "krige", "mlp", "lstm", "tcn"]
    P = np.stack([np.stack([preds[s][m] for m in names], 0) for s in todo], 0)  # (n_sites, M, T)
    np.savez(os.path.join(out_dir, f"preds_{tag}.npz"), P=P, names=names, sites=np.array(todo))
    rows = []
    for a, s in enumerate(todo):
        z = S[:, s]
        for b, m in enumerate(names):
            sc = score(z, P[a, b])
            ok = np.isfinite(z) & np.isfinite(P[a, b])
            sc["v"] = float(np.var(P[a, b][ok]) / np.var(z[ok])) if ok.sum() > 2 else np.nan
            rows.append(dict(site=s, method=m, archive=str(d["archive"][s]), proxy=str(d["proxy"][s]), **sc))
    print(f"\n  {'method':9s} median r   median v   median SS   SS > 0   (tree sites | other sites: median SS)")
    tree = d["archive"][todo] == "tree"
    for m in names:
        rr = [x for x in rows if x["method"] == m]
        r = np.array([x["r"] for x in rr]); v = np.array([x["v"] for x in rr]); ss = np.array([x["ss"] for x in rr])
        print(f"  {m:9s} {np.nanmedian(r):+.3f}    {np.nanmedian(v):.3f}     {np.nanmedian(ss):+.3f}     "
              f"{np.nanmean(ss > 0):.0%}      {np.nanmedian(ss[tree]):+.3f} | {np.nanmedian(ss[~tree]):+.3f}")
    print(f"  kriging: median sum of weights {np.median([info[s]['krige_wsum'] for s in todo]):.2f}")
    import csv
    with open(os.path.join(out_dir, f"scores_{tag}.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"  wrote {out_dir}/scores_{tag}.csv, preds_{tag}.npz  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/cfr-data/pages2kv2.json")
    ap.add_argument("--window", type=int, nargs=2, default=[1500, 2001])
    ap.add_argument("--sites", type=int, nargs=2, help="half-open [a, b) of site indices (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(tuple(a.window), a.path, a.sites, a.workers, a.out)
