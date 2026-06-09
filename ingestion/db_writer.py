"""
Database writer — inserts parsed Argo DataFrames into PostgreSQL.
Supports batch upsert and idempotent re-runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings


def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


def init_schema(schema_path: Path | None = None) -> None:
    """Create all tables from schema.sql."""
    if schema_path is None:
        schema_path = Path(__file__).parents[1] / "database" / "schema.sql"
    engine = get_engine()
    sql = schema_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.success("Schema initialized.")


def upsert_float(engine, row: dict) -> None:
    with engine.begin() as conn:
        stmt = text("""
            INSERT INTO floats (float_id, platform_number, dac, ocean_basin,
                                wmo_inst_type, positioning_sys)
            VALUES (:float_id, :platform_number, :dac, :ocean_basin,
                    :wmo_inst_type, :positioning_sys)
            ON CONFLICT (float_id) DO NOTHING
        """)
        conn.execute(stmt, row)


def upsert_profiles(engine, df: pd.DataFrame) -> dict[tuple, int]:
    """
    Insert profiles and return {(float_id, cycle_number, direction): profile_id}.
    """
    if df.empty:
        return {}

    id_map: dict[tuple, int] = {}
    with engine.begin() as conn:
        for _, row in df.iterrows():
            stmt = text("""
                INSERT INTO profiles
                    (float_id, cycle_number, juld, latitude, longitude,
                     position_qc, direction, data_mode, source_file)
                VALUES
                    (:float_id, :cycle_number, :juld, :latitude, :longitude,
                     :position_qc, :direction, :data_mode, :source_file)
                ON CONFLICT (float_id, cycle_number, direction) DO UPDATE SET
                    juld        = EXCLUDED.juld,
                    latitude    = EXCLUDED.latitude,
                    longitude   = EXCLUDED.longitude,
                    source_file = EXCLUDED.source_file
                RETURNING profile_id
            """)
            result = conn.execute(stmt, row.to_dict())
            profile_id = result.scalar()
            key = (row["float_id"], int(row["cycle_number"]), row.get("direction", "A"))
            id_map[key] = profile_id
    return id_map


def insert_measurements(engine, df: pd.DataFrame, id_map: dict) -> int:
    """Bulk-insert measurements mapped to real profile_ids."""
    if df.empty:
        return 0

    profiles_meta = _get_profiles_meta(engine, list(id_map.keys()))
    rows = []
    for _, row in df.iterrows():
        pidx = int(row["profile_idx"])
        if pidx >= len(profiles_meta):
            continue
        pm = profiles_meta[pidx]
        key = (pm["float_id"], pm["cycle_number"], pm["direction"])
        profile_id = id_map.get(key)
        if profile_id is None:
            continue
        rows.append({
            "profile_id":    profile_id,
            "pressure":      row.get("pressure"),
            "temperature":   row.get("temperature"),
            "salinity":      row.get("salinity"),
            "temp_adjusted": row.get("temp_adjusted"),
            "psal_adjusted": row.get("psal_adjusted"),
            "pres_adjusted": row.get("pres_adjusted"),
            "temp_qc":       row.get("temp_qc", "0")[:1],
            "psal_qc":       row.get("psal_qc", "0")[:1],
            "pres_qc":       row.get("pres_qc", "0")[:1],
        })

    if not rows:
        return 0

    BATCH = 5000
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            conn.execute(
                text("""
                    INSERT INTO measurements
                        (profile_id, pressure, temperature, salinity,
                         temp_adjusted, psal_adjusted, pres_adjusted,
                         temp_qc, psal_qc, pres_qc)
                    VALUES
                        (:profile_id, :pressure, :temperature, :salinity,
                         :temp_adjusted, :psal_adjusted, :pres_adjusted,
                         :temp_qc, :psal_qc, :pres_qc)
                """),
                chunk,
            )
    logger.debug(f"Inserted {len(rows)} measurement rows")
    return len(rows)


def log_ingestion(engine, file_path: str, float_id: str,
                  profiles_count: int, status: str, error: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ingestion_log (file_path, float_id, profiles_count, status, error_msg)
            VALUES (:file_path, :float_id, :profiles_count, :status, :error_msg)
            ON CONFLICT (file_path) DO UPDATE SET
                status = EXCLUDED.status,
                error_msg = EXCLUDED.error_msg,
                ingested_at = NOW()
        """), {
            "file_path":      file_path,
            "float_id":       float_id,
            "profiles_count": profiles_count,
            "status":         status,
            "error_msg":      error,
        })


def already_ingested(engine, file_path: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT status FROM ingestion_log WHERE file_path = :p AND status = 'success'"
        ), {"p": file_path})
        return result.fetchone() is not None


def _get_profiles_meta(engine, keys: list[tuple]) -> list[dict]:
    """Re-fetch profile metadata in insertion order for idx mapping."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT float_id, cycle_number, direction FROM profiles ORDER BY profile_id"
        ))
        return [dict(r._mapping) for r in result]
