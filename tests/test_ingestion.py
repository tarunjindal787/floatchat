"""
Tests for the ingestion pipeline — NetCDF parser and DB writer.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── NetCDF Parser tests ───────────────────────────────────────────────────────

def _make_mock_dataset(n_prof: int = 3, n_levels: int = 10):
    """Create a minimal mock xarray Dataset mimicking Argo NetCDF structure."""
    import xarray as xr

    data = {
        "PLATFORM_NUMBER": (["N_PROF", "STRING8"],
            np.array([[c for c in f"1234567{' '*(7)}"][:8]
                      for _ in range(n_prof)], dtype="U1")),
        "JULD":         (["N_PROF"],   np.full(n_prof, 25567.0)),   # ~2020
        "LATITUDE":     (["N_PROF"],   np.linspace(10, 20, n_prof)),
        "LONGITUDE":    (["N_PROF"],   np.linspace(60, 70, n_prof)),
        "CYCLE_NUMBER": (["N_PROF"],   np.arange(1, n_prof + 1)),
        "POSITION_QC":  (["N_PROF"],   np.array(["1"] * n_prof, dtype="U1")),
        "DIRECTION":    (["N_PROF"],   np.array(["A"] * n_prof, dtype="U1")),
        "DATA_MODE":    (["N_PROF"],   np.array(["R"] * n_prof, dtype="U1")),
        "PRES":  (["N_PROF", "N_LEVELS"], np.tile(np.linspace(0, 2000, n_levels), (n_prof, 1))),
        "TEMP":  (["N_PROF", "N_LEVELS"], np.random.uniform(5, 30, (n_prof, n_levels))),
        "PSAL":  (["N_PROF", "N_LEVELS"], np.random.uniform(34, 37, (n_prof, n_levels))),
        "TEMP_QC": (["N_PROF", "N_LEVELS"], np.full((n_prof, n_levels), "1", dtype="U1")),
        "PSAL_QC": (["N_PROF", "N_LEVELS"], np.full((n_prof, n_levels), "1", dtype="U1")),
        "PRES_QC": (["N_PROF", "N_LEVELS"], np.full((n_prof, n_levels), "1", dtype="U1")),
    }
    ds = xr.Dataset(data)
    return ds


def test_parse_produces_correct_keys():
    """parse_argo_file should return dict with 4 expected keys."""
    from ingestion.netcdf_parser import parse_argo_file

    with tempfile.NamedTemporaryFile(suffix="_prof.nc", delete=False) as f:
        tmp = Path(f.name)

    ds = _make_mock_dataset()
    ds.to_netcdf(tmp)

    result = parse_argo_file(tmp)
    tmp.unlink(missing_ok=True)

    assert set(result.keys()) >= {"float_meta", "profiles", "measurements"}


def test_profiles_count():
    """profiles DataFrame should have n_prof rows."""
    from ingestion.netcdf_parser import parse_argo_file

    n_prof = 5
    with tempfile.NamedTemporaryFile(suffix="_prof.nc", delete=False) as f:
        tmp = Path(f.name)
    ds = _make_mock_dataset(n_prof=n_prof)
    ds.to_netcdf(tmp)
    result = parse_argo_file(tmp)
    tmp.unlink(missing_ok=True)

    assert len(result["profiles"]) == n_prof


def test_measurements_have_pressure():
    """measurements DataFrame must contain a 'pressure' column."""
    from ingestion.netcdf_parser import parse_argo_file

    with tempfile.NamedTemporaryFile(suffix="_prof.nc", delete=False) as f:
        tmp = Path(f.name)
    ds = _make_mock_dataset()
    ds.to_netcdf(tmp)
    result = parse_argo_file(tmp)
    tmp.unlink(missing_ok=True)

    assert "pressure" in result["measurements"].columns


def test_infer_ocean():
    """_infer_ocean should return 'Indian Ocean' for coordinates in the basin."""
    from ingestion.netcdf_parser import _infer_ocean
    import xarray as xr

    ds = xr.Dataset({
        "LATITUDE":  (["N_PROF"], np.array([10.0, 15.0])),
        "LONGITUDE": (["N_PROF"], np.array([65.0, 70.0])),
    })
    result = _infer_ocean(ds)
    assert "Indian" in result


def test_float_meta_has_float_id():
    """float_meta should have a float_id column."""
    from ingestion.netcdf_parser import parse_argo_file

    with tempfile.NamedTemporaryFile(suffix="_prof.nc", delete=False) as f:
        tmp = Path(f.name)
    ds = _make_mock_dataset()
    ds.to_netcdf(tmp)
    result = parse_argo_file(tmp)
    tmp.unlink(missing_ok=True)

    assert "float_id" in result["float_meta"].columns
    assert len(result["float_meta"]) == 1


# ── DB Writer tests ───────────────────────────────────────────────────────────

def test_already_ingested_false(tmp_path):
    """already_ingested should return False for unknown files."""
    from unittest.mock import MagicMock, patch
    from ingestion.db_writer import already_ingested

    mock_engine = MagicMock()
    mock_conn   = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = None
    mock_conn.execute.return_value    = mock_result
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__  = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    result = already_ingested(mock_engine, "/fake/path.nc")
    assert result is False


def test_build_profile_summary():
    """build_profile_summary should produce a non-empty string."""
    from ingestion.vector_indexer import build_profile_summary

    row = {
        "float_id":      "6902742",
        "cycle_number":  45,
        "juld":          "2023-03-15",
        "latitude":      12.3,
        "longitude":     68.1,
        "ocean_basin":   "Indian Ocean",
        "mean_temp":     26.5,
        "mean_salinity": 36.1,
        "max_pressure":  2000.0,
    }
    summary = build_profile_summary(row)
    assert "6902742" in summary
    assert "Indian Ocean" in summary
    assert len(summary) > 50
