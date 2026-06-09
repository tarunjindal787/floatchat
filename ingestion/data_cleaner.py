"""
Data Cleaner — QC flag filtering, unit normalization, and anomaly removal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# Physical oceanographic value bounds for basic range checks
BOUNDS = {
    "temperature": (-2.5, 40.0),   # °C
    "salinity":    (2.0,  42.0),   # PSU
    "pressure":    (0.0,  12000.0),# dbar
    "doxy":        (0.0,  600.0),  # µmol/kg
    "chla":        (0.0,  50.0),   # mg/m³
    "nitrate":     (0.0,  200.0),  # µmol/kg
    "ph_in_situ":  (7.0,  9.0),
    "bbp700":      (0.0,  1.0),
}

GOOD_QC = {"1", "2"}


def clean_measurements(df: pd.DataFrame,
                        apply_qc: bool = True,
                        apply_range: bool = True) -> pd.DataFrame:
    """
    Apply QC flag filtering and range checks to a measurements DataFrame.

    Parameters
    ----------
    df          : DataFrame with columns: pressure, temperature, salinity,
                  temp_qc, psal_qc, pres_qc
    apply_qc    : Drop rows where QC flags are bad (not in GOOD_QC)
    apply_range : Replace out-of-bounds values with NaN

    Returns cleaned DataFrame.
    """
    original_len = len(df)

    if apply_qc:
        df = _filter_qc(df, "temperature", "temp_qc")
        df = _filter_qc(df, "salinity",    "psal_qc")
        df = _filter_qc(df, "pressure",    "pres_qc")

    if apply_range:
        for col, (lo, hi) in BOUNDS.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                n_bad = mask.sum()
                if n_bad > 0:
                    logger.debug(f"Range check: {n_bad} bad {col} values → NaN")
                    df.loc[mask, col] = np.nan

    # Drop rows with no usable data
    key_cols = [c for c in ["temperature", "salinity"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols, how="all")

    logger.debug(
        f"clean_measurements: {original_len} → {len(df)} rows "
        f"({'QC+range' if apply_qc and apply_range else 'QC' if apply_qc else 'range'})"
    )
    return df.reset_index(drop=True)


def clean_bgc(df: pd.DataFrame, apply_qc: bool = True,
              apply_range: bool = True) -> pd.DataFrame:
    """Apply QC and range cleaning to BGC DataFrame."""
    original_len = len(df)

    bgc_qc_pairs = [
        ("doxy",      "doxy_qc"),
        ("chla",      "chla_qc"),
        ("nitrate",   "nitrate_qc"),
        ("ph_in_situ","ph_qc"),
        ("bbp700",    "bbp700_qc"),
    ]

    if apply_qc:
        for col, qc_col in bgc_qc_pairs:
            df = _filter_qc(df, col, qc_col)

    if apply_range:
        for col, (lo, hi) in BOUNDS.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                df.loc[mask, col] = np.nan

    logger.debug(f"clean_bgc: {original_len} → {len(df)} rows")
    return df.reset_index(drop=True)


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize longitudes to [-180, 180] range.
    Remove profiles with invalid coordinates.
    """
    if "longitude" in df.columns:
        df["longitude"] = ((df["longitude"] + 180) % 360) - 180
    if "latitude" in df.columns:
        bad_lat = (df["latitude"] < -90) | (df["latitude"] > 90)
        bad_lon = (df["longitude"] < -180) | (df["longitude"] > 180)
        n_bad = (bad_lat | bad_lon).sum()
        if n_bad:
            logger.warning(f"Removing {n_bad} profiles with invalid coordinates")
            df = df[~(bad_lat | bad_lon)]
    return df.reset_index(drop=True)


def _filter_qc(df: pd.DataFrame, col: str, qc_col: str) -> pd.DataFrame:
    """Set col to NaN where qc_col is not in GOOD_QC."""
    if col not in df.columns or qc_col not in df.columns:
        return df
    bad = ~df[qc_col].astype(str).isin(GOOD_QC)
    df.loc[bad, col] = np.nan
    return df
