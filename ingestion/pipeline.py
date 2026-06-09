"""
Main ingestion pipeline — orchestrates NetCDF parsing → DB → Vector index.
Run:  python -m ingestion.pipeline --input data/raw/ --region indian_ocean
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from sqlalchemy import text

from config.settings import get_settings
from ingestion.netcdf_parser import parse_argo_file, scan_directory
from ingestion.db_writer import (
    get_engine, init_schema, upsert_float,
    upsert_profiles, insert_measurements,
    log_ingestion, already_ingested,
)
from ingestion.vector_indexer import index_profiles


@click.command()
@click.option("--input",  "-i", "input_dir",  default="data/raw",  help="Directory of NetCDF files")
@click.option("--region", "-r", default=None,  help="Filter by ocean basin substring")
@click.option("--init-schema", "do_init",   is_flag=True, default=False)
@click.option("--rebuild-index", "rebuild",  is_flag=True, default=False)
@click.option("--skip-existing", "skip",     is_flag=True, default=True)
def run_pipeline(input_dir: str, region: str | None,
                 do_init: bool, rebuild: bool, skip: bool) -> None:

    logger.remove()
    logger.add(sys.stderr, level=get_settings().app_log_level)
    logger.add("logs/ingestion.log", rotation="50 MB", retention="30 days")

    engine = get_engine()

    if do_init:
        init_schema()

    root = Path(input_dir)
    nc_files = list(scan_directory(root))
    logger.info(f"Found {len(nc_files)} NetCDF profile files under {root}")

    profile_records_for_index: list[dict] = []

    for nc_path in nc_files:
        str_path = str(nc_path)

        if skip and already_ingested(engine, str_path):
            logger.debug(f"Skipping already-ingested: {nc_path.name}")
            continue

        try:
            result = parse_argo_file(nc_path)
            if not result:
                continue

            float_meta_df  = result["float_meta"]
            profiles_df    = result["profiles"]
            meas_df        = result["measurements"]

            # Filter by region if requested
            if region:
                ocean = float_meta_df["ocean_basin"].iloc[0]
                if region.lower() not in ocean.lower():
                    continue

            # Write float
            upsert_float(engine, float_meta_df.iloc[0].to_dict())

            # Write profiles; get back {(fid, cycle, dir) -> profile_id}
            id_map = upsert_profiles(engine, profiles_df)

            # Write measurements
            insert_measurements(engine, meas_df, id_map)

            # Collect for vector index
            stats = _fetch_profile_stats(engine, list(id_map.values()))
            profile_records_for_index.extend(stats)

            log_ingestion(engine, str_path,
                          float_meta_df["float_id"].iloc[0],
                          len(profiles_df), "success")

            logger.success(f"  ✓ {nc_path.name}: {len(profiles_df)} profiles")

        except Exception as exc:
            logger.error(f"  ✗ {nc_path.name}: {exc}")
            log_ingestion(engine, str_path, "unknown", 0, "error", str(exc))

    # Vector index
    if profile_records_for_index:
        logger.info(f"Indexing {len(profile_records_for_index)} profiles into ChromaDB …")
        index_profiles(profile_records_for_index)

    logger.success("Pipeline complete.")


def _fetch_profile_stats(engine, profile_ids: list[int]) -> list[dict]:
    if not profile_ids:
        return []
    id_list = ",".join(map(str, profile_ids))
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT profile_id, float_id, juld, latitude, longitude,
                   cycle_number, data_mode,
                   AVG(m.temperature)  AS mean_temp,
                   AVG(m.salinity)     AS mean_salinity,
                   MAX(m.pressure)     AS max_pressure
            FROM profiles p
            LEFT JOIN measurements m USING (profile_id)
            WHERE p.profile_id IN ({id_list})
            GROUP BY profile_id, float_id, juld, latitude, longitude,
                     cycle_number, data_mode
        """))
        return [dict(r._mapping) for r in result]


if __name__ == "__main__":
    run_pipeline()
