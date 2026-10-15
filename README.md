# paleo-sparse

Sparse and learned-dictionary priors for reconstructing climate fields from paleoclimate proxies.

## The idea

Proxy-based field reconstruction is a compressive-sensing problem. In row-vector form, with the field x (1 × N grid cells), proxies y (1 × M sites), sensing operator Φ (N × M) and a dictionary Ψ (K × N):

    y = x Φ + ε,   x = c Ψ   ⇒   y = c (Ψ Φ) + ε

Standard reconstructions (offline EnKF as in the Last Millennium Reanalysis, PHYDA, RegEM, BARCAST) effectively use a Gaussian, ℓ2 prior on a low-rank Ψ (EOFs or an ensemble). This project tests a **sparse prior in a learned, overcomplete Ψ** (dictionary learning or a sparse autoencoder trained on climate-model output), recovered with LASSO or OMP:

    ĉ = argmin_c ½‖y − c Ψ Φ‖² + λ‖c‖₁,   x̂ = ĉ Ψ

Evaluation follows the standard pseudoproxy protocol: a climate-model run is the truth, sampled at real proxy sites with noise, then reconstructed and scored (CE, RE, r) against the truth.

## Status

- `src/lme_loader.py`: opens a CESM1 Last Millennium Ensemble monthly netCDF time series and returns annual-mean anomalies X (T × N), lat/lon and √cos(lat) weights. Run `python src/lme_loader.py --selftest`.
- `src/eof_pseudoproxy.py`: weighted EOFs, pseudoproxy sampling at proxy sites, an EOF-regression (ℓ2) baseline and CE/correlation scoring. The demo runs on a synthetic field at the real PAGES 2k sites (`figures/`).
- Next: real LME data; then the sparse method and the sparse-vs-Gaussian comparison.

## Data (not committed)

**CESM1 LME** (NCAR GDEX d651058). One member's monthly TREFHT is about 0.47 GB:

```bash
B=https://tds.gdex.ucar.edu/thredds/fileServer/files/d651058/CESM-CAM5-LME/atm/proc/tseries/monthly/TREFHT
curl -O $B/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.085001-184912.nc
curl -O $B/b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.185001-200512.nc
python src/lme_loader.py b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.*.nc --var TREFHT
```

If those URLs fail, use the dataset's access page: https://gdex.ucar.edu/datasets/d651058/dataaccess/

**PAGES 2k temperature v2.0 site list** (via the cfr data repo):

```bash
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/fzhu2e/cfr-data data/cfr-data
git -C data/cfr-data checkout HEAD -- pages2kv2.json
python src/eof_pseudoproxy.py --pages2k data/cfr-data/pages2kv2.json
```

## Environment

```bash
conda env create -f environment.yml
conda activate paleo
```

## Key references

- Tardif et al. 2019, Last Millennium Reanalysis: https://cp.copernicus.org/articles/15/1251/2019/
- Guillot, Rajaratnam & Emile-Geay 2015, GraphEM: https://arxiv.org/abs/1309.6702
- Wang et al. 2014, pseudoproxy protocol: https://cp.copernicus.org/articles/10/1/2014/cp-10-1-2014.pdf
- Zhu et al. 2024, cfr: https://gmd.copernicus.org/articles/17/3409/2024/
- Valley & Shaw 2011, OMP/NLS for off-grid sparse recovery: https://doi.org/10.5772/25842

## People

T. Justin Shaw ([@wyojustin](https://github.com/wyojustin)) and Andrew Flaim.
