"""
Pytest configuration and shared fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_profiles_df() -> pd.DataFrame:
    """Sample profiles DataFrame for testing."""
    return pd.DataFrame({
        "profile_id":   [1, 2, 3],
        "float_id":     ["6902742", "6902742", "2902733"],
        "cycle_number": [1, 2, 1],
        "juld":         pd.to_datetime(["2023-01-15", "2023-02-10", "2023-03-05"]),
        "latitude":     [12.3, 13.1, 18.5],
        "longitude":    [68.1, 68.5, 72.3],
        "data_mode":    ["R", "R", "A"],
        "direction":    ["A", "A", "A"],
    })


@pytest.fixture
def sample_measurements_df() -> pd.DataFrame:
    """Sample CTD measurements DataFrame for testing."""
    n = 50
    profiles = np.repeat([1, 2], [25, 25])
    pressures = np.tile(np.linspace(0, 2000, 25), 2)
    return pd.DataFrame({
        "profile_id":  profiles,
        "pressure":    pressures,
        "temperature": np.random.uniform(4, 30, n),
        "salinity":    np.random.uniform(34, 37, n),
        "temp_qc":     ["1"] * n,
        "psal_qc":     ["1"] * n,
    })


@pytest.fixture
def sample_bgc_df() -> pd.DataFrame:
    """Sample BGC DataFrame for testing."""
    n = 30
    return pd.DataFrame({
        "profile_id": [1] * n,
        "pressure":   np.linspace(0, 1000, n),
        "doxy":       np.random.uniform(150, 280, n),
        "chla":       np.random.uniform(0, 2, n),
        "nitrate":    np.random.uniform(0, 30, n),
        "bbp700":     np.random.uniform(0, 0.01, n),
    })


@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM responses."""
    def _make(text: str):
        from unittest.mock import MagicMock, AsyncMock
        client = MagicMock()
        client.complete_sync.return_value = text
        client.complete = AsyncMock(return_value=text)
        return client
    return _make
