"""
data_loader.py
----------------
Locates and loads the retail interaction dataset from the local `data/` folder.

The loader is intentionally defensive: it works with either CSV or Excel files,
and it auto-detects common column-name variants so the rest of the pipeline can
rely on a small set of standardized column names regardless of how the raw
file happens to label things.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Project root = one level up from this file's parent (src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Candidate column names (case-sensitive variants a real-world export might use)
COLUMN_CANDIDATES = {
    "user_id": ["user_id", "User ID", "user", "customer_id", "Customer ID", "userid"],
    "product_id": ["product_id", "Product ID", "product", "item_id", "productid"],
    "interaction": ["interaction_type", "interaction", "Interaction", "event_type", "action"],
    "timestamp": ["timestamp", "Timestamp", "date", "Date", "event_time"],
    "product_name": ["product_name", "Product Name", "item_name", "name"],
    "category": ["category", "Category", "product_category", "Product Category"],
    "brand": ["brand", "Brand"],
    "price": ["price", "Price"],
}


def find_dataset_file(data_dir: Path = DATA_DIR) -> Optional[Path]:
    """
    Search the data directory for a usable dataset file.

    Looks for CSV files first (preferring one literally named 'dataset.csv'),
    then falls back to any .csv, .xlsx, or .xls file found in the folder.

    Returns:
        Path to the dataset file, or None if nothing suitable was found.
    """
    if not data_dir.exists():
        return None

    preferred = data_dir / "dataset.csv"
    if preferred.exists():
        return preferred

    # Any CSV file
    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    # Any Excel file
    excel_files = sorted(data_dir.glob("*.xlsx")) + sorted(data_dir.glob("*.xls"))
    if excel_files:
        return excel_files[0]

    return None


def _map_columns(df: pd.DataFrame) -> dict:
    """
    Build a mapping from standardized column names to the actual column
    names present in the dataframe, based on COLUMN_CANDIDATES.
    """
    found = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for standard_name, candidates in COLUMN_CANDIDATES.items():
        match = None
        for candidate in candidates:
            if candidate in df.columns:
                match = candidate
                break
            if candidate.lower().strip() in lower_cols:
                match = lower_cols[candidate.lower().strip()]
                break
        if match is not None:
            found[standard_name] = match

    return found


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[Optional[pd.DataFrame], dict, Optional[str]]:
    """
    Load the dataset and standardize known columns.

    Returns:
        (dataframe, column_map, error_message)
        - dataframe is None and error_message is set if loading failed.
        - column_map maps standardized names (e.g. 'user_id') to the actual
          column name found in the file, for columns that were detected.
    """
    file_path = find_dataset_file(data_dir)

    if file_path is None:
        return None, {}, (
            "Dataset not found. Please place the downloaded CSV file "
            f"inside the '{data_dir}' folder (see README for instructions)."
        )

    try:
        if file_path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001 - surface any read error to the caller
        return None, {}, f"Failed to read dataset file '{file_path.name}': {exc}"

    if df.empty:
        return None, {}, f"Dataset file '{file_path.name}' was found but contains no rows."

    column_map = _map_columns(df)

    # The pipeline cannot function without at least user, product, and interaction info.
    required = ["user_id", "product_id", "interaction"]
    missing_required = [r for r in required if r not in column_map]
    if missing_required:
        return None, column_map, (
            "Dataset is missing required column(s): "
            f"{', '.join(missing_required)}. Found columns: {list(df.columns)}"
        )

    return df, column_map, None


if __name__ == "__main__":
    df, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    print(f"Loaded dataset with shape: {df.shape}")
    print("Detected column mapping:")
    for k, v in colmap.items():
        print(f"  {k:15s} -> {v}")
