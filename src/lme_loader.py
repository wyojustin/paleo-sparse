"""
CESM1 Last Millennium Ensemble (LME) loader -> X (T x N) anomaly matrix.

Conventions
  * Row-vector: X[t, j] is year t, grid cell j. One row per time step.
  * Ranges are half-open [a, b), like Python slices. ref=(1951, 1981) means
    years 1951..1980 inclusive.
  * Never fail silently. Missing packages, files, variables or metadata raise
    errors naming what is missing. Recoverable gaps ask quit/continue when
    interactive, abort when not, and are summarized in field.notes.

Real files (verified on NCAR GDEX THREDDS, dataset d651058), one member's
monthly 2 m air temperature is two files:
  b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.085001-184912.nc   (~408 MB)
  b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT.185001-200512.nc   (~64 MB)
Swap TREFHT for TS (surface temp) or PRECT (total precip, m/s).

CESM quirk handled here: monthly h0 time stamps sit at the END of the
averaging interval (January 0850 is stamped 0850-02-01). We assign each month
to the midpoint of its time bounds, so annual means use the right months.

Usage
  python src/lme_loader.py FILE1.nc [FILE2.nc ...] --var TREFHT
  python src/lme_loader.py --selftest        # synthetic CESM-like files
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import xarray as xr
except ImportError as e:  # fail loudly, name what is missing
    raise ImportError(
        "lme_loader needs 'xarray' (and 'netCDF4' + 'cftime' to read CESM "
        "files with years < 1678 on a noleap calendar).\n"
        "  conda install -c conda-forge xarray netcdf4 cftime"
    ) from e
for _mod in ("netCDF4", "cftime"):
    try:
        __import__(_mod)
    except ImportError as e:
        raise ImportError(
            f"lme_loader needs '{_mod}'. conda install -c conda-forge {_mod.lower()}"
        ) from e


# --------------------------------------------------------------------------
# loud-failure helper
# --------------------------------------------------------------------------
FORCE_NONINTERACTIVE = False      # the self-test sets this so its planted problems always abort


def ask_continue(msg: str) -> None:
    """Interactive: offer quit/continue. Non-interactive: abort."""
    if not FORCE_NONINTERACTIVE and sys.stdin is not None and sys.stdin.isatty():
        ans = input(f"\nWARNING: {msg}\n  [q]uit or [c]ontinue? ").strip().lower()
        if ans.startswith("c"):
            return
        raise SystemExit(f"Aborted by user: {msg}")
    raise RuntimeError(f"Non-interactive run, aborting: {msg}")


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
@dataclass
class Field:
    X: np.ndarray            # (T, N) anomalies, unweighted, float64
    years: np.ndarray        # (T,) int
    lat: np.ndarray          # (nlat,)
    lon: np.ndarray          # (nlon,)
    mask: np.ndarray         # (nlat, nlon) bool, True = column kept in X
    cell_lat: np.ndarray     # (N,) latitude of each column of X
    cell_lon: np.ndarray     # (N,)
    w: np.ndarray            # (N,) sqrt(cos(lat)) EOF weights
    clim: np.ndarray         # (N,) reference-period mean that was removed
    var: str
    units: str
    notes: list = field(default_factory=list)

    def to_grid(self, row: np.ndarray) -> np.ndarray:
        """(N,) row -> (nlat, nlon) map with NaN outside the mask."""
        g = np.full(self.mask.shape, np.nan)
        g[self.mask] = row
        return g


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------
def open_var(path: str, var: str) -> xr.DataArray:
    """Open one CESM monthly time-series file and return var with
    a float 'tmid' coordinate (fractional year at interval midpoint)."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"LME file not found: {path}")
    coders = getattr(xr, "coders", None)
    if coders is not None and hasattr(coders, "CFDatetimeCoder"):   # xarray >= 2025.01
        ds = xr.open_dataset(path, decode_times=coders.CFDatetimeCoder(use_cftime=True))
    else:                                                           # older xarray
        ds = xr.open_dataset(path, use_cftime=True)
    if var not in ds:
        raise KeyError(f"Variable '{var}' not in {path}. Data variables: "
                       f"{sorted(ds.data_vars)}")
    da = ds[var]
    for d in ("time", "lat", "lon"):
        if d not in da.dims:
            raise KeyError(f"'{var}' in {path} lacks dimension '{d}' (dims={da.dims})")

    bname = ds["time"].attrs.get("bounds", "time_bnds")
    if bname in ds:
        b = ds[bname].values                       # (T, 2) cftime objects
        y0 = np.array([t.year + (t.dayofyr - 1) / 365.0 for t in b[:, 0]])
        y1 = np.array([t.year + (t.dayofyr - 1) / 365.0 for t in b[:, 1]])
        tmid = 0.5 * (y0 + y1)
        dt = y1 - y0                                # month length (yr), weights
    else:
        ask_continue(f"{path}: no time bounds ('{bname}'). Will shift CESM "
                     "end-of-interval stamps back by 1/24 yr and weight "
                     "months equally.")
        t = ds["time"].values
        tmid = np.array([x.year + (x.dayofyr - 1) / 365.0 for x in t]) - 1 / 24
        dt = np.full(tmid.shape, 1 / 12)
    da = da.assign_coords(tmid=("time", tmid), dt=("time", dt))
    da.attrs.setdefault("units", "?")
    return da


def annual_mean(da: xr.DataArray, min_months: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Month-length-weighted calendar-year means. Returns (years, A[T, nlat, nlon]).
    Incomplete years (< min_months) are dropped and reported, never hidden."""
    yr = np.floor(da["tmid"].values).astype(int)
    dt = da["dt"].values
    vals = da.values.astype(np.float64)                # (Tm, nlat, nlon)
    uyears = np.unique(yr)
    out, keep, dropped = [], [], []
    for y in uyears:
        sel = yr == y
        if sel.sum() < min_months:
            dropped.append((int(y), int(sel.sum())))
            continue
        wt = dt[sel] / dt[sel].sum()
        out.append(np.tensordot(wt, vals[sel], axes=(0, 0)))
        keep.append(y)
    if dropped:
        print(f"[annual_mean] dropped incomplete years (year, n_months): {dropped}",
              file=sys.stderr)
    if not out:
        raise ValueError("No complete years found.")
    return np.array(keep), np.stack(out)


def load_lme(paths, var: str = "TREFHT", ref=(1951, 1981), years=None,
             lat_range=None, min_months: int = 12) -> Field:
    """Load one member (one or more files, e.g. 0850-1849 + 1850-2005).

    ref        half-open [y0, y1) reference period for anomalies.
    years      optional half-open [y0, y1) to keep.
    lat_range  optional half-open [lat0, lat1) band to keep (e.g. (-60, 90)).
    """
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("load_lme: no file paths given.")
    notes = []
    yrs_all, A_all, lat, lon, units = [], [], None, None, None
    for p in paths:                       # annualize per file: bounded memory
        da = open_var(p, var)
        y, A = annual_mean(da, min_months)
        if lat is None:
            lat, lon, units = da["lat"].values, da["lon"].values, da.attrs["units"]
        elif not (np.allclose(lat, da["lat"].values) and np.allclose(lon, da["lon"].values)):
            raise ValueError(f"Grid in {p} differs from first file.")
        yrs_all.append(y)
        A_all.append(A)
    years_v = np.concatenate(yrs_all)
    A = np.concatenate(A_all)
    order = np.argsort(years_v, kind="stable")
    years_v, A = years_v[order], A[order]
    if np.any(np.diff(years_v) == 0):
        raise ValueError(f"Duplicate years across files: "
                         f"{np.unique(years_v[1:][np.diff(years_v) == 0])}")
    gaps = np.where(np.diff(years_v) > 1)[0]
    if gaps.size:
        msg = f"year gaps after {years_v[gaps].tolist()}"
        notes.append(msg)
        ask_continue(msg)

    if years is not None:
        s = (years_v >= years[0]) & (years_v < years[1])
        if not s.any():
            raise ValueError(f"No data in years [{years[0]}, {years[1]})")
        years_v, A = years_v[s], A[s]

    r = (years_v >= ref[0]) & (years_v < ref[1])
    if not r.any():
        raise ValueError(f"Reference period [{ref[0]}, {ref[1]}) has no data; "
                         f"available [{years_v[0]}, {years_v[-1] + 1})")
    if r.sum() < ref[1] - ref[0]:
        msg = f"reference period [{ref[0]}, {ref[1]}) only has {r.sum()} years"
        notes.append(msg)
        ask_continue(msg)

    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    mask = np.all(np.isfinite(A), axis=0)
    n_nonfinite = int((~mask).sum())
    if n_nonfinite:
        notes.append(f"{n_nonfinite} grid cells had non-finite values and were masked")
    if lat_range is not None:
        mask &= (LAT >= lat_range[0]) & (LAT < lat_range[1])

    X = A[:, mask]                                  # (T, N) row-major time
    clim = X[r].mean(axis=0)
    X = X - clim
    cell_lat, cell_lon = LAT[mask], LON[mask]
    w = np.sqrt(np.clip(np.cos(np.deg2rad(cell_lat)), 0.0, None))
    for n in notes:
        print(f"[load_lme] NOTE: {n}", file=sys.stderr)
    return Field(X=X, years=years_v, lat=lat, lon=lon, mask=mask,
                 cell_lat=cell_lat, cell_lon=cell_lon, w=w, clim=clim,
                 var=var, units=units, notes=notes)


# --------------------------------------------------------------------------
# self-test on synthetic CESM-like files
# --------------------------------------------------------------------------
def _write_fake_cesm(path, y0, y1, lat, lon, rng, var="TREFHT"):
    """Monthly file for years [y0, y1), noleap calendar, CESM-style
    end-of-interval stamps and time_bnds. Each year's value = 280 + year/1000
    + seasonal cycle, so the annual mean is known exactly."""
    import cftime
    ml = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    nm = (y1 - y0) * 12
    t0, t1, data = [], [], []
    seas = 10 * np.sin(2 * np.pi * (np.arange(12) + 0.5) / 12)
    seas -= (seas * ml).sum() / 365.0              # zero annual mean (weighted)
    for k in range(nm):
        y, m = y0 + k // 12, k % 12 + 1
        a = cftime.DatetimeNoLeap(y, m, 1)
        b = cftime.DatetimeNoLeap(y + (m == 12), m % 12 + 1, 1)
        t0.append(a); t1.append(b)
        data.append(np.full((lat.size, lon.size), 280 + y / 1000 + seas[m - 1])
                    + 1e-3 * rng.standard_normal((lat.size, lon.size)))
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), np.array(data, np.float32), {"units": "K"}),
         "time_bnds": (("time", "nbnd"), np.array([t0, t1]).T)},
        coords={"time": ("time", np.array(t1), {"bounds": "time_bnds"}),
                "lat": lat, "lon": lon})
    enc = {"time": {"units": "days since 0850-01-01 00:00:00", "calendar": "noleap"},
           "time_bnds": {"units": "days since 0850-01-01 00:00:00", "calendar": "noleap"}}
    ds.to_netcdf(path, encoding=enc)


def selftest():
    import os
    import tempfile
    global FORCE_NONINTERACTIVE
    FORCE_NONINTERACTIVE = True                    # planted gap must abort, even in a terminal
    rng = np.random.default_rng(0)
    lat = np.linspace(-90, 90, 19)                 # coarse stand-in for 96 x 144
    lon = np.arange(0, 360, 20.0)
    d = tempfile.mkdtemp(prefix="lme_selftest_")
    case = "b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TREFHT"
    f1 = os.path.join(d, f"{case}.085001-086012.nc")
    f2 = os.path.join(d, f"{case}.185001-186012.nc")
    _write_fake_cesm(f1, 850, 861, lat, lon, rng)
    _write_fake_cesm(f2, 1850, 1861, lat, lon, rng)
    try:
        load_lme([f1, f2], "TREFHT", ref=(850, 861))
        raise AssertionError("gap between files should have aborted (non-interactive)")
    except RuntimeError as e:
        print(f"[selftest] gap correctly flagged: {e}")
    f = load_lme([f2], "TREFHT", ref=(1850, 1861))
    assert f.years.tolist() == list(range(1850, 1861)), f.years
    expect = 280 + f.years / 1000.0
    expect = expect - expect.mean()
    err = np.abs(f.X - expect[:, None]).max()
    print(f"[selftest] X shape {f.X.shape}, years [{f.years[0]}, {f.years[-1] + 1}), "
          f"max |annual-mean error| = {err:.2e} K (end-of-interval stamps handled)")
    assert err < 1e-3, err
    assert f.w.shape == (f.X.shape[1],) and np.all(f.w >= 0)
    assert np.isnan(f.to_grid(f.X[0])).sum() == 0
    try:
        load_lme([f2], "TS")
    except KeyError as e:
        print(f"[selftest] missing variable correctly reported: {e}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--var", default="TREFHT")
    ap.add_argument("--ref", nargs=2, type=int, default=[1951, 1981],
                    help="half-open reference period [y0, y1)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.files:
        selftest()
    else:
        f = load_lme(a.files, a.var, ref=tuple(a.ref))
        print(f"{f.var} [{f.units}]  X: {f.X.shape} (T x N)  years "
              f"[{f.years[0]}, {f.years[-1] + 1})  grid {f.mask.shape}, "
              f"kept {f.mask.sum()} cells")
