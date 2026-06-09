"""
Script to download sample Indian Ocean Argo float data from GDAC.
Run: python scripts/download_sample_data.py --n-floats 20
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import click
import requests
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "raw"

# Argo GDAC HTTP mirrors
GDAC_MIRRORS = [
    "https://data-argo.ifremer.fr/dac/",
    "https://usgodae.org/ftp/outgoing/argo/dac/",
]

# Indian Ocean DACs with significant coverage
INDIAN_OCEAN_DACS = ["incois", "csio", "bodc", "coriolis", "meds"]

# Known Indian Ocean float IDs (WMO numbers) for quick demo
SAMPLE_FLOAT_IDS = [
    "2902733", "2902734", "2902735", "2902736", "2902737",  # INCOIS
    "6901760", "6901761", "6901762",                         # CORIOLIS
    "5905988", "5905989", "5905990",                         # MEDS
    "4902480", "4902481",                                    # BODC
    "2903586", "2903587", "2903588",                         # CSIO
]


@click.command()
@click.option("--n-floats", "-n", default=10, help="Number of floats to download")
@click.option("--float-ids", "-f", default=None, help="Comma-separated WMO float IDs")
@click.option("--output-dir", "-o", default=str(DATA_DIR))
def download(n_floats: int, float_ids: str | None, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if float_ids:
        ids = [f.strip() for f in float_ids.split(",")]
    else:
        ids = SAMPLE_FLOAT_IDS[:n_floats]

    logger.info(f"Downloading {len(ids)} floats to {out}")

    for fid in tqdm(ids, desc="Downloading floats"):
        _download_float(fid, out)
        time.sleep(0.5)  # be polite to servers

    logger.success(f"Download complete. Files in: {out}")


def _download_float(float_id: str, out_dir: Path) -> None:
    """Try to download the latest profile file for a float from GDAC mirrors."""
    for mirror in GDAC_MIRRORS:
        dac = _guess_dac(float_id)
        url = f"{mirror}{dac}/{float_id}/{float_id}_prof.nc"
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200:
                dest = out_dir / f"{float_id}_prof.nc"
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"  ✓ {float_id} → {dest.name} ({dest.stat().st_size/1024:.0f} KB)")
                return
        except Exception as exc:
            logger.debug(f"  Mirror {mirror} failed for {float_id}: {exc}")

    logger.warning(f"  ✗ Could not download float {float_id}")


def _guess_dac(float_id: str) -> str:
    """Guess DAC from float ID prefix (rough heuristic)."""
    prefix = int(float_id[:2])
    if prefix in [29]:
        return "incois"
    elif prefix in [69]:
        return "coriolis"
    elif prefix in [59]:
        return "meds"
    elif prefix in [49]:
        return "bodc"
    return "coriolis"


if __name__ == "__main__":
    download()
