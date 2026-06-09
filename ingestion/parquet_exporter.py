"""
Parquet exporter — saves processed Argo data partitioned by ocean_basin/year/month.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text

from ingestion.db_writer import get_engine


def export_profiles_to_parquet(output_dir: Path = Path("data/processed")) -> None:
    """Export the profiles + measurements joined table to partitioned Parquet files."""
    engine = get_engine()
    output_dir.mkdir(parents=True, exist_ok=True)

    query = text("""
        SELECT
            p.profile_id, p.float_id, p.cycle_number,
            p.juld, p.latitude, p.longitude,
            p.data_mode, p.direction,
            f.ocean_basin,
            m.pressure, m.temperature, m.salinity,
            m.temp_adjusted, m.psal_adjusted,
            m.temp_qc, m.psal_qc
        FROM profiles p
        JOIN floats f USING (float_id)
        JOIN measurements m USING (profile_id)
        ORDER BY p.juld, p.float_id, p.cycle_number, m.pressure
    """)

    logger.info("Exporting profiles+measurements to Parquet …")
    df = pd.read_sql(query, engine)
    df["year"]  = pd.to_datetime(df["juld"]).dt.year
    df["month"] = pd.to_datetime(df["juld"]).dt.month

    for (basin, year, month), grp in df.groupby(["ocean_basin", "year", "month"]):
        path = output_dir / basin.replace(" ", "_") / str(year) / f"{month:02d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        grp.drop(columns=["ocean_basin", "year", "month"]).to_parquet(path, index=False)
        logger.debug(f"  {path} ({len(grp):,} rows)")

    logger.success(f"Parquet export complete → {output_dir}")


def load_parquet(ocean_basin: str = "Indian_Ocean",
                 year: int | None = None,
                 month: int | None = None,
                 base_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Load Parquet files with optional year/month filtering."""
    pattern = base_dir / ocean_basin
    if year:
        pattern = pattern / str(year)
        if month:
            pattern = pattern / f"{month:02d}.parquet"
            return pd.read_parquet(pattern)
    files = list(pattern.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {pattern}")
    return pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
