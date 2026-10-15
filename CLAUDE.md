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

## Status

- `src/lme_loader.py`: CESM1 LME monthly netCDF → annual-mean anomalies X (T × N), sqrt(cos lat) weights. `--selftest` passes.
- `src/eof_pseudoproxy.py`: weighted EOFs, pseudoproxy sampling at PAGES 2k sites, EOF-regression ℓ2 baseline, CE/corr metrics. `--lme FILES` uses real LME as truth.
- **First real result** (member 002 TREFHT, 692 PAGES 2k sites in 355 cells, SNR 0.5, calibrate [1850, 2006), validate [850, 1850)):

  | r EOFs | GMT corr | GMT CE | cell CE (area mean) |
  |---|---|---|---|
  | 1 | 0.771 | 0.543 | 0.090 |
  | 4 | 0.864 | 0.716 | 0.213 |
  | 16 | 0.892 | 0.781 | 0.314 |

  EOF spectrum decays slowly (EOF 0 = 22%): compressible, not sparse. Poor skill in the Southern Ocean; r = 4 imposes spurious polar bands in volcanic years (e.g. 1259, Samalas). Volcanic dips (1259, ~1453, ~1600, 1815) are captured in the global mean.
- **Next single step**: test whether the LME field is sparse in a learned dictionary. On held-out years, compare reconstruction error at equal sparsity L: top-L EOFs vs L atoms of a learned overcomplete dictionary. No proxies yet; this only tests the premise.
- Then: staircase proxy networks, cfr baselines (EnKF, GraphEM), LASSO/OMP with learned Ψ, phase transition.

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
