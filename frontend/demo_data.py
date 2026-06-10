"""
Demo mode helpers — loads synthetic Indian Ocean data when no DB is connected.
Used by Streamlit Cloud where no PostgreSQL is available on first launch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_db_available() -> bool:
    """Return True if PostgreSQL is reachable."""
    try:
        from ingestion.db_writer import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def make_demo_profiles(n: int = 200) -> pd.DataFrame:
    """Generate realistic-looking Indian Ocean profile locations."""
    np.random.seed(42)
    float_ids = [f"690{i:04d}" for i in range(1, 21)]
    rows = []
    for fid in float_ids:
        n_cycles = n // len(float_ids)
        base_lat = np.random.uniform(5, 25)
        base_lon = np.random.uniform(55, 95)
        for c in range(1, n_cycles + 1):
            rows.append({
                "profile_id":    len(rows) + 1,
                "float_id":      fid,
                "cycle_number":  c,
                "juld":          pd.Timestamp("2023-01-01") + pd.Timedelta(days=10 * c),
                "latitude":      base_lat  + np.random.uniform(-2, 2),
                "longitude":     base_lon  + np.random.uniform(-2, 2),
                "mean_temp":     np.random.uniform(20, 30),
                "mean_salinity": np.random.uniform(34.5, 36.5),
                "max_pressure":  np.random.uniform(500, 2000),
                "ocean_basin":   "Indian Ocean",
                "data_mode":     "R",
            })
    return pd.DataFrame(rows)


def make_demo_ctd(profile_id: int = 1) -> pd.DataFrame:
    """Synthetic CTD profile."""
    n = 60
    pressure = np.linspace(0, 2000, n)
    temp     = 28 * np.exp(-pressure / 800) + np.random.normal(0, 0.2, n)
    salinity = 35.0 + 1.5 * (1 - np.exp(-pressure / 300)) + np.random.normal(0, 0.05, n)
    return pd.DataFrame({
        "profile_id": profile_id,
        "pressure":   pressure,
        "temperature": temp,
        "salinity":    salinity,
        "temp_qc":    "1",
        "psal_qc":    "1",
    })


def make_demo_bgc(profile_id: int = 1) -> pd.DataFrame:
    """Synthetic BGC profile."""
    n = 40
    pressure = np.linspace(0, 1000, n)
    doxy     = 250 - 80  * (1 - np.exp(-pressure / 200)) + np.random.normal(0, 2, n)
    chla     = 2.0 * np.exp(-((pressure - 50) ** 2) / (2 * 30 ** 2)) + np.random.normal(0, 0.05, n)
    nitrate  = 25 * (1 - np.exp(-pressure / 150)) + np.random.normal(0, 0.5, n)
    return pd.DataFrame({
        "profile_id": profile_id,
        "pressure":   pressure,
        "doxy":       np.clip(doxy, 0, None),
        "chla":       np.clip(chla, 0, None),
        "nitrate":    np.clip(nitrate, 0, None),
    })


def make_demo_trajectory(float_id: str = "6900001") -> pd.DataFrame:
    """Synthetic float trajectory (random walk)."""
    np.random.seed(int(float_id[-4:]) % 100)
    n = 50
    lats = np.cumsum(np.random.normal(0, 0.3, n)) + 15.0
    lons = np.cumsum(np.random.normal(0, 0.3, n)) + 65.0
    dates = pd.date_range("2023-01-01", periods=n, freq="10D")
    return pd.DataFrame({
        "float_id":     float_id,
        "cycle_number": range(1, n + 1),
        "juld":         dates,
        "latitude":     lats,
        "longitude":    lons,
    })
