# CLAUDE.md: context for working in this repo

## Who and why

- **Justin Shaw** (repo owner, @wyojustin): senior aerospace engineer, pure-math MS, published in compressive sensing (CS) and reservoir computing. Co-author with George Valley of OMP/NLS, an OMP variant that refines off-grid continuous parameters with nonlinear least squares (Valley & Shaw 2011, doi:10.5772/25842).
- **Andrew Flaim** (collaborator): climate-science postdoc at Rice (WashU PhD on hydroclimate and water isotopes in proxies and models). Has HPC compute.
- The project ties to Justin's sparse-autoencoder (SAE) interpretability work: an SAE is amortized LASSO, and the same tool can learn an overcomplete dictionary of climate patterns.

## The idea

Proxy-based climate field reconstruction as compressive sensing, row-vector convention:

    y = x Φ + ε,  x = c Ψ  ⇒  y = c (Ψ Φ) + ε
    ĉ = argmin ½‖y − c Ψ Φ‖² + λ‖c‖₁,  x̂ = ĉ Ψ

- x: field, 1 × N grid cells. y: proxies, 1 × M sites. Φ: sensing (site selection, later proxy system models). Ψ: dictionary, K × N.
- Every standard method (LMR offline EnKF, PHYDA, RegEM, BARCAST, CCA/PCR) is effectively a Gaussian / ℓ2 prior on a low-rank Ψ (EOFs or an ensemble).
- Proposal: a learned **overcomplete** Ψ (dictionary learning or an SAE on CESM-LME output) with an ℓ1 prior (LASSO, OMP), compared against ℓ2 methods in pseudoproxy experiments, including a phase-transition analysis (proxy count m vs active atoms k).
- Prior-art check (2026-10-15, web search, not yet a Scholar sweep): nothing found using a learned sparse dictionary as the reconstruction prior. Closest: GraphEM (Guillot, Rajaratnam & Emile-Geay 2015; ℓ1 on the precision matrix, Gaussian prior on the field; ~43% lower MSE than RegEM-TTLS). SAEs on AI weather models exist (GraphCast, SONNY, ClimaX) but interpret network activations, never physical fields.

## Status: measurement-only track (active since 2026-10-15)

No CESM or other forward model as training data; reanalysis only as a comparison. Learn from PAGES 2k proxies alone. Validation: leave-one-site-out over 418 sites (509 annual records, each z-scored within the window; co-located records averaged into one site). Score per site: r, v = var(ẑ)/var(z), SS = 1 − MSE/var (0 = predict the mean); SS = 2r√v − v − bias² exactly.

- `src/proxy_data.py`: loads all 692 records (fixes 193 reversed and 15 mixed-order records, drops 838 NaN pairs), annual matrix Y, overview figure. 509 annual records, 61 sub-annual (corals, ice), 122 coarser. Seasonal share of variance in sub-annual corals: median 0.34.
- `src/proxy_baselines.py`: nearest neighbor and spherical Delaunay (convex hull of unit vectors, barycentric weights). Both negative median SS: copying a neighbor gives SS = 2r − 1, and r ≈ 0.3. Delaunay is better mainly through accidental shrinkage (v = 0.68).
- Correlogram (all pairs): r ≈ 0.5 at < 50 km (nugget: half a tree record is site-specific), a fast drop by 100 km, then decay over about 400 to 700 km, and a +0.03 tail. Non-tree pairs r ≲ 0.1 at all distances.
- `src/proxy_learned.py` (full leave-one-out, about 23 min per window on 16 cores) and `src/proxy_summary.py`: ridge, kriging (two-scale correlogram plus nugget), MLP, bidirectional LSTM, temporal CNN; K = 8 neighbors plus proxy class. Median SS:

  | window | nearest | Delaunay | ridge | krige | MLP | LSTM | TCN |
  |---|---|---|---|---|---|---|---|
  | [1500, 2001) | −0.35 | −0.09 | +0.160 | +0.148 | +0.158 | +0.163 | +0.163 |
  | [1000, 2001) | −0.39 | −0.15 | +0.144 | +0.135 | +0.138 | +0.139 | +0.136 |

  All learned models beat both baselines at about 80 % of sites. Paired per site, kriging is best (beats ridge at 67 to 70 % of sites, beats LSTM at 63 to 65 %; bootstrap CIs exclude 0); the neural nets add only +0.003 to +0.006 over ridge. Tree sites median SS about +0.2 to +0.25; non-tree sites ≈ 0.
- `src/proxy_probe.py`: the neural nets converge to kriging: total weight about 0.7 (shrinkage), weights decay with distance like the correlogram, no sign flips learned between proxy classes, temporal kernel almost a delta at τ = 0 (LSTM: small −0.025 at τ = −1).
- Not yet decided (Justin's call): which direction to deepen.

## Status: CESM pseudoproxy track (paused)

- `src/lme_loader.py`: CESM1 LME monthly netCDF → annual-mean anomalies X (T × N), sqrt(cos lat) weights. `--selftest` passes.
- `src/eof_pseudoproxy.py`: weighted EOFs, pseudoproxy sampling at PAGES 2k sites, EOF-regression ℓ2 baseline, CE/corr metrics. `--lme FILES` uses real LME as truth.
- **First real result** (member 002 TREFHT, 692 PAGES 2k sites in 355 cells, SNR 0.5, calibrate [1850, 2006), validate [850, 1850)):

  | r EOFs | GMT corr | GMT CE | cell CE (area mean) |
  |---|---|---|---|
  | 1 | 0.771 | 0.543 | 0.090 |
  | 4 | 0.864 | 0.716 | 0.213 |
  | 16 | 0.892 | 0.781 | 0.314 |

  EOF spectrum decays slowly (EOF 0 = 22%): compressible, not sparse. Poor skill in the Southern Ocean; r = 4 imposes spurious polar bands in volcanic years (e.g. 1259, Samalas). Volcanic dips (1259, ~1453, ~1600, 1815) are captured in the global mean.
- `src/sparsity_test.py`: **premise test** (no proxies). Held-out unexplained fraction at equal sparsity L: EOF top-L, EOF best-L per year, analog dictionary (normalized training years, OMP), K-SVD (K ∈ {64, 256}, OMP). `--selftest` passes (recovers a planted dictionary). Result (member 002 TREFHT, decade-block split, mean over 5 block phases):

  | L | EOF top-L | EOF best-L | analog | K-SVD 256 |
  |---|---|---|---|---|
  | 1 | 0.782 | 0.709 | 0.615 | 0.597 |
  | 4 | 0.578 | 0.454 | 0.387 | 0.392 |
  | 16 | 0.323 | 0.234 | 0.206 | 0.214 |

  Premise holds: at fixed L, sparse codes in a 256-atom dictionary leave about 2/3 of the EOF top-L error (L = 4: 0.39 vs 0.58), consistently across all 5 phases and on the forced-warming split (train [850, 1850), test [1850, 2006)). About half the gain already comes from adaptive selection in the EOF basis. K-SVD does not beat the untrained analog dictionary for L ≥ 4: with 926 training years it overfits (train error is about half the test error). Caveat: an L-sparse code also carries log2 C(K, L) bits of support.
- `src/proxy_recon.py`: **proxy step**. Same pseudoproxies (692 PAGES 2k, 355 cells), same prior years, noise variance from calibration residuals. ℓ2: `gauss` (Kalman / offline EnKF with the prior years as ensemble, full rank), `gauss_r` (rank r), `eofreg_r` (earlier baseline). ℓ1: OMP (whole L path, plus chi2 discrepancy stop) and LASSO (FISTA, α sweep), both in analog and K-SVD dictionaries. Designs: `cal` (prior [1850, 2006), test [850, 1850)) and `blocks` (prior = all but held-out decades in [850, 1850)). Validation cell CE (area mean); sparse values are the best over L or α (**oracle**):

  | SNR | design | gauss | gauss_16 | eofreg_16 | best OMP | best LASSO | chi2 OMP |
  |---|---|---|---|---|---|---|---|
  | 0.5 | cal | **0.414** | 0.337 | 0.314 | 0.141 | 0.350 | -0.14 |
  | 0.5 | blocks | **0.497** | 0.399 | 0.271 | 0.196 | 0.415 | -0.19 |
  | 2 | cal | 0.489 | 0.368 | 0.399 | 0.356 | **0.506** (α at sweep edge, 71 nnz) | 0.35 |
  | 2 | blocks | **0.663** | 0.429 | 0.362 | 0.434 | 0.606 | 0.40 |

  **Negative result.** The approximation gain from the premise test does not carry over to recovery. The full-rank Gaussian (Kalman) wins in every case except one oracle-tuned, nearly dense LASSO. Likely reasons: (1) OMP's least-squares coefficients are unshrunk, so at SNR 0.5 they amplify noise (skill peaks at L ≈ 4, then CE goes negative); (2) support selection from 355 noisy cells is unreliable; (3) the chi2 stop overfits (σ² is estimated from 156 years, and the low SNR leaves little signal energy above the noise floor). Side finding: full-rank `gauss` beats every truncated EOF method by about 0.08 to 0.25 CE, so the earlier `eofreg_r` baseline was weak.
- **Open next steps** (pick one): (a) shrink on the support: OMP/LASSO support, then Gaussian-posterior (MMSE) coefficients with prior variances from training codes (sparse Bayesian, closer to a fair ℓ1 vs ℓ2 test); (b) more LME members for training; (c) SNR × proxy-count phase diagram for gauss vs sparse, to find where, if anywhere, sparse wins.
- Then: cfr baselines (EnKF, GraphEM).

## Data (never committed; `data/` is gitignored)

- CESM1 LME (NCAR GDEX d651058), member 002 TREFHT, already in `data/` on Justin's Mac:
  `b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.{085001-184912,185001-200512}.nc` (96 × 144 grid, 1,156 years).
  Download base: `https://tds.gdex.ucar.edu/thredds/fileServer/files/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/TREFHT/`
- PAGES 2k sites: `data/cfr-data/pages2kv2.json` (from github.com/fzhu2e/cfr-data).
- The `cfr` package (Zhu et al. 2024, Python ≥ 3.13) has pseudoproxy tools, proxy models, LMR-style EnKF and GraphEM: use it for baselines.

## How Justin works (follow these)

- **Pace**: one small step at a time; explain, then wait for him before moving on. Getting ahead wastes time. Understand first, optimize later.
- **Math**: row-vector convention (X is T × N, one row per time step; products are X @ W). Zero-based, half-open ranges [a, b) in code and prose, never inclusive "a to b".
- **Code**: NumPy + matplotlib. Never fail silently: name what is missing; interactively offer quit/continue; abort when non-interactive; summarize anything skipped.
- **Files**: generated documents and figures start with the date, `yyyy-mm-dd_name.ext`. Plots: save, then show at the end.
- **Plots**: no bar charts; use dot plots or lines.
- **Env**: conda (miniforge), env `paleo` from `environment.yml`.
- **Style**: concise, direct, include the math, push back when warranted. No em-dashes.
- **Email/outreach**: draft only; never send without Justin approving the exact text.
- Git commits: end messages with `Co-Authored-By: Claude <noreply@anthropic.com>`.
