"""
Data export component — CSV, NetCDF, and ASCII table downloads.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from tabulate import tabulate


def export_buttons(df: pd.DataFrame, key_prefix: str = "export") -> None:
    """Render CSV, NetCDF, and ASCII export buttons for a DataFrame."""
    if df is None or df.empty:
        st.caption("No data to export.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name="argo_data.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
            use_container_width=True,
        )

    with col2:
        nc_bytes = _df_to_netcdf(df)
        st.download_button(
            label="⬇ Download NetCDF",
            data=nc_bytes,
            file_name="argo_data.nc",
            mime="application/octet-stream",
            key=f"{key_prefix}_nc",
            use_container_width=True,
        )

    with col3:
        ascii_bytes = _df_to_ascii(df).encode("utf-8")
        st.download_button(
            label="⬇ Download ASCII",
            data=ascii_bytes,
            file_name="argo_data.txt",
            mime="text/plain",
            key=f"{key_prefix}_ascii",
            use_container_width=True,
        )


def _df_to_netcdf(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to NetCDF bytes."""
    try:
        ds = xr.Dataset.from_dataframe(df)
        # Add CF-compliant metadata
        ds.attrs.update({
            "title":       "ARGO Float Data — FloatChat Export",
            "institution": "FloatChat System",
            "Conventions": "CF-1.8",
            "source":      "Argo float measurements",
        })
        # Add variable attributes for known columns
        _var_attrs = {
            "temperature": {"long_name": "Sea Water Temperature", "units": "degree_Celsius", "standard_name": "sea_water_temperature"},
            "salinity":    {"long_name": "Practical Salinity",    "units": "PSU",            "standard_name": "sea_water_practical_salinity"},
            "pressure":    {"long_name": "Sea Water Pressure",    "units": "decibar",        "standard_name": "sea_water_pressure"},
            "latitude":    {"long_name": "Latitude",              "units": "degrees_north",  "standard_name": "latitude"},
            "longitude":   {"long_name": "Longitude",             "units": "degrees_east",   "standard_name": "longitude"},
            "doxy":        {"long_name": "Dissolved Oxygen",      "units": "micromol/kg",    "standard_name": "moles_of_oxygen_per_unit_mass_in_sea_water"},
            "chla":        {"long_name": "Chlorophyll-a",         "units": "mg/m3"},
        }
        for var, attrs in _var_attrs.items():
            if var in ds.data_vars:
                ds[var].attrs.update(attrs)

        buf = io.BytesIO()
        ds.to_netcdf(buf, engine="scipy")
        return buf.getvalue()
    except Exception:
        # Fallback: return raw CSV bytes labelled as NetCDF
        return df.to_csv(index=False).encode("utf-8")


def _df_to_ascii(df: pd.DataFrame, max_rows: int = 500) -> str:
    """Convert DataFrame to a fixed-width ASCII table."""
    sample = df.head(max_rows)
    header = (
        "# FloatChat ARGO Data Export\n"
        f"# Rows: {len(df)} (showing first {min(max_rows, len(df))})\n"
        f"# Columns: {', '.join(df.columns)}\n"
        "#\n"
    )
    table = tabulate(sample, headers="keys", tablefmt="simple", showindex=False,
                     floatfmt=".4f")
    return header + table


def show_data_table(df: pd.DataFrame, max_rows: int = 200) -> None:
    """Render a styled Streamlit dataframe with row count."""
    if df is None or df.empty:
        st.info("No data returned.")
        return

    st.caption(f"**{len(df):,}** rows × **{len(df.columns)}** columns")
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        height=300,
    )
    if len(df) > max_rows:
        st.caption(f"*Showing first {max_rows} rows. Download to see all.*")
