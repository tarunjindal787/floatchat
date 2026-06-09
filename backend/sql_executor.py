"""
Safe SQL executor — runs validated SELECT queries against PostgreSQL.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text

from config.settings import get_settings
from ingestion.db_writer import get_engine


def execute_sql(sql: str, limit: int | None = None) -> pd.DataFrame:
    """
    Execute a pre-validated SQL query and return a DataFrame.
    Automatically appends LIMIT if not already present.
    """
    s = get_settings()
    cap = limit or s.max_sql_rows

    clean = sql.strip().rstrip(";")
    if "limit" not in clean.lower():
        clean = f"{clean} LIMIT {cap}"

    engine = get_engine()
    logger.debug(f"Executing SQL:\n{clean}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(clean), conn)
        logger.info(f"Query returned {len(df)} rows")
        return df
    except Exception as exc:
        logger.error(f"SQL execution error: {exc}")
        raise RuntimeError(f"Database query failed: {exc}") from exc


def get_float_trajectory(float_id: str) -> pd.DataFrame:
    """Return ordered lat/lon/time series for a float."""
    sql = text("""
        SELECT cycle_number, juld, latitude, longitude
        FROM profiles
        WHERE float_id = :fid AND latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY juld
    """)
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"fid": float_id})


def get_profile_data(profile_id: int) -> pd.DataFrame:
    """Return full CTD profile for a single profile_id."""
    sql = text("""
        SELECT pressure, temperature, salinity, temp_qc, psal_qc
        FROM measurements
        WHERE profile_id = :pid AND pressure IS NOT NULL
        ORDER BY pressure
    """)
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"pid": profile_id})


def get_bgc_profile(profile_id: int) -> pd.DataFrame:
    """Return BGC profile for a single profile_id."""
    sql = text("""
        SELECT pressure, doxy, chla, nitrate, ph_in_situ, bbp700
        FROM bgc_data
        WHERE profile_id = :pid
        ORDER BY pressure
    """)
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"pid": profile_id})


def get_nearest_floats(lat: float, lon: float, top_n: int = 5) -> pd.DataFrame:
    """Find the N most recent profiles nearest to a given lat/lon."""
    sql = text("""
        SELECT
            p.float_id, p.profile_id, p.juld,
            p.latitude, p.longitude,
            ROUND(
                CAST(
                    ST_Distance(
                        ST_SetSRID(ST_MakePoint(p.longitude, p.latitude), 4326)::geography,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                    ) / 1000.0
                AS NUMERIC), 2
            ) AS distance_km
        FROM profiles p
        WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
        ORDER BY
            ST_Distance(
                ST_SetSRID(ST_MakePoint(p.longitude, p.latitude), 4326)::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            )
        LIMIT :n
    """)
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"lat": lat, "lon": lon, "n": top_n})
