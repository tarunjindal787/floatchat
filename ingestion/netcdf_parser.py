"""
ARGO NetCDF Parser
Reads core and BGC Argo NetCDF files and returns structured DataFrames.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


# ── QC Flags considered "good" ───────────────────────────────────────────────
GOOD_QC_FLAGS = {"1", "2"}

# ── ARGO variable mappings ────────────────────────────────────────────────────
CORE_VARS = {
    "PRES":  ("pressure",    "PRES_QC",  "PRES_ADJUSTED"),
    "TEMP":  ("temperature", "TEMP_QC",  "TEMP_ADJUSTED"),
    "PSAL":  ("salinity",    "PSAL_QC",  "PSAL_ADJUSTED"),
}

BGC_VARS = {
    "DOXY":          ("doxy",         "DOXY_QC"),
    "CHLA":          ("chla",         "CHLA_QC"),
    "NITRATE":       ("nitrate",      "NITRATE_QC"),
    "PH_IN_SITU_TOTAL": ("ph_in_situ", "PH_IN_SITU_TOTAL_QC"),
    "BBP700":        ("bbp700",       "BBP700_QC"),
    "IRRADIANCE_380": ("irradiance_380", None),
    "IRRADIANCE_412": ("irradiance_412", None),
    "IRRADIANCE_490": ("irradiance_490", None),
}


def _mask_bad_qc(values: np.ndarray, qc_array: np.ndarray) -> np.ndarray:
    """Replace values with NaN where QC flag is not in GOOD_QC_FLAGS."""
    masked = values.astype(float).copy()
    qc_chars = np.array(
        [str(int(q)) if not np.isnan(float(q)) else "0" for q in qc_array.flat]
    ).reshape(qc_array.shape)
    bad = ~np.isin(qc_chars, list(GOOD_QC_FLAGS))
    masked[bad] = np.nan
    return masked


def _decode_char_array(arr: np.ndarray) -> list[str]:
    """Decode byte/char arrays from NetCDF into Python strings."""
    if arr.dtype.kind in ("S", "U"):
        return ["".join(row.astype(str)).strip() for row in arr]
    return [str(v).strip() for v in arr]


def parse_argo_file(nc_path: Path) -> dict[str, pd.DataFrame]:
    """
    Parse a single Argo NetCDF file.

    Returns a dict with keys:
        'float_meta'   -> pd.DataFrame (1 row per float)
        'profiles'     -> pd.DataFrame (1 row per profile)
        'measurements' -> pd.DataFrame (1 row per pressure level)
        'bgc_data'     -> pd.DataFrame (BGC levels, may be empty)
    """
    logger.info(f"Parsing: {nc_path.name}")
    ds = xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True)

    n_prof = ds.dims.get("N_PROF", 0)
    if n_prof == 0:
        logger.warning(f"No profiles in {nc_path.name}")
        ds.close()
        return {}

    # ── Float-level metadata ─────────────────────────────────────────────────
    platform_numbers = _decode_char_array(ds["PLATFORM_NUMBER"].values)
    float_id = platform_numbers[0] if platform_numbers else nc_path.stem

    dac = nc_path.parts[-3] if len(nc_path.parts) >= 3 else "unknown"

    float_meta = pd.DataFrame([{
        "float_id":        float_id,
        "platform_number": float_id,
        "dac":             dac,
        "ocean_basin":     _infer_ocean(ds),
        "wmo_inst_type":   _get_scalar_char(ds, "WMO_INST_TYPE"),
        "positioning_sys": _get_scalar_char(ds, "POSITIONING_SYSTEM"),
    }])

    # ── Profile-level metadata ───────────────────────────────────────────────
    juld_raw = ds["JULD"].values  # days since 1950-01-01
    juld_ref = np.datetime64("1950-01-01", "D")
    juld_ts = [
        pd.Timestamp(str(juld_ref + int(j))) if not np.isnan(float(j)) else pd.NaT
        for j in juld_raw
    ]

    cycle_numbers = ds["CYCLE_NUMBER"].values.astype(int)
    latitudes     = ds["LATITUDE"].values.astype(float)
    longitudes    = ds["LONGITUDE"].values.astype(float)
    pos_qc        = _decode_char_array(ds["POSITION_QC"].values) if "POSITION_QC" in ds else ["0"] * n_prof
    direction     = _decode_char_array(ds["DIRECTION"].values) if "DIRECTION" in ds else ["A"] * n_prof
    data_mode     = _decode_char_array(ds["DATA_MODE"].values) if "DATA_MODE" in ds else ["R"] * n_prof

    profiles_rows = []
    for i in range(n_prof):
        profiles_rows.append({
            "float_id":     float_id,
            "cycle_number": int(cycle_numbers[i]),
            "juld":         juld_ts[i],
            "latitude":     float(latitudes[i]) if not np.isnan(latitudes[i]) else None,
            "longitude":    float(longitudes[i]) if not np.isnan(longitudes[i]) else None,
            "position_qc":  pos_qc[i][0] if pos_qc[i] else "0",
            "direction":    direction[i][0] if direction[i] else "A",
            "data_mode":    data_mode[i][0] if data_mode[i] else "R",
            "source_file":  str(nc_path),
        })

    profiles_df = pd.DataFrame(profiles_rows)

    # ── Measurement data (N_PROF × N_LEVELS) ─────────────────────────────────
    meas_rows = []
    for i in range(n_prof):
        pres_vals = _get_var_values(ds, "PRES", i)
        temp_vals = _get_var_values(ds, "TEMP", i)
        psal_vals = _get_var_values(ds, "PSAL", i)
        temp_adj  = _get_var_values(ds, "TEMP_ADJUSTED", i)
        psal_adj  = _get_var_values(ds, "PSAL_ADJUSTED", i)
        pres_adj  = _get_var_values(ds, "PRES_ADJUSTED", i)
        temp_qc   = _get_qc_values(ds, "TEMP_QC", i)
        psal_qc   = _get_qc_values(ds, "PSAL_QC", i)
        pres_qc   = _get_qc_values(ds, "PRES_QC", i)

        n_levels = len(pres_vals)
        for j in range(n_levels):
            if np.isnan(pres_vals[j]) and np.isnan(temp_vals[j]):
                continue
            meas_rows.append({
                "profile_idx":  i,
                "pressure":     _safe_float(pres_vals[j]),
                "temperature":  _safe_float(temp_vals[j]),
                "salinity":     _safe_float(psal_vals[j]),
                "temp_adjusted": _safe_float(temp_adj[j]) if len(temp_adj) > j else None,
                "psal_adjusted": _safe_float(psal_adj[j]) if len(psal_adj) > j else None,
                "pres_adjusted": _safe_float(pres_adj[j]) if len(pres_adj) > j else None,
                "temp_qc":      temp_qc[j] if len(temp_qc) > j else "0",
                "psal_qc":      psal_qc[j] if len(psal_qc) > j else "0",
                "pres_qc":      pres_qc[j] if len(pres_qc) > j else "0",
            })

    meas_df = pd.DataFrame(meas_rows)

    # ── BGC data ──────────────────────────────────────────────────────────────
    bgc_df = _parse_bgc(ds, n_prof)

    ds.close()

    return {
        "float_meta":   float_meta,
        "profiles":     profiles_df,
        "measurements": meas_df,
        "bgc_data":     bgc_df,
    }


def _parse_bgc(ds: xr.Dataset, n_prof: int) -> pd.DataFrame:
    rows = []
    for name, (col, qc_name) in BGC_VARS.items():
        if name not in ds.variables:
            continue
        for i in range(n_prof):
            vals = _get_var_values(ds, name, i)
            qcs  = _get_qc_values(ds, qc_name, i) if qc_name and qc_name in ds else ["1"] * len(vals)
            for j, v in enumerate(vals):
                if np.isnan(v):
                    continue
                rows.append({
                    "profile_idx": i,
                    "bgc_var":     col,
                    "pressure":    None,  # joined separately from PRES
                    "value":       float(v),
                    "qc":          qcs[j] if j < len(qcs) else "0",
                })

    if not rows:
        return pd.DataFrame()

    # Pivot to wide format per profile/level
    bgc_raw = pd.DataFrame(rows)
    return bgc_raw


def scan_directory(root: Path) -> Iterator[Path]:
    """Yield all .nc files under a directory."""
    for p in sorted(root.rglob("*.nc")):
        if "_prof" in p.name or p.name.endswith("prof.nc"):
            yield p


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_var_values(ds: xr.Dataset, var: str, prof_idx: int) -> np.ndarray:
    if var not in ds.variables:
        return np.array([np.nan])
    arr = ds[var].values
    if arr.ndim == 2:
        return arr[prof_idx].astype(float)
    return arr.astype(float)


def _get_qc_values(ds: xr.Dataset, var: str, prof_idx: int) -> list[str]:
    if var not in ds.variables:
        return []
    arr = ds[var].values
    if arr.ndim == 2:
        row = arr[prof_idx]
    else:
        row = arr
    return [str(v).strip() if str(v).strip() else "0" for v in row]


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _get_scalar_char(ds: xr.Dataset, var: str) -> str:
    if var not in ds.variables:
        return ""
    arr = ds[var].values
    if arr.ndim == 2:
        return "".join(arr[0].astype(str)).strip()
    return "".join(arr.astype(str)).strip()


def _infer_ocean(ds: xr.Dataset) -> str:
    """Infer ocean basin from mean longitude/latitude."""
    lats = ds["LATITUDE"].values.astype(float)
    lons = ds["LONGITUDE"].values.astype(float)
    valid_lat = lats[~np.isnan(lats)]
    valid_lon = lons[~np.isnan(lons)]
    if len(valid_lat) == 0:
        return "unknown"
    mean_lat = float(np.mean(valid_lat))
    mean_lon = float(np.mean(valid_lon))
    # Normalize longitude to 0–360
    mean_lon_360 = mean_lon % 360
    if 20 <= mean_lon_360 <= 100 and -60 <= mean_lat <= 30:
        return "Indian Ocean"
    elif mean_lon_360 < 20 or mean_lon_360 > 290:
        return "Atlantic Ocean"
    elif 100 <= mean_lon_360 <= 290:
        return "Pacific Ocean"
    elif mean_lat < -60:
        return "Southern Ocean"
    elif mean_lat > 65:
        return "Arctic Ocean"
    return "Unknown"
