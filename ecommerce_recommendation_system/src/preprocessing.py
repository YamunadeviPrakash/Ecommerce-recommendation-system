"""
preprocessing.py
-----------------
Cleans the raw interaction dataframe and produces a standardized dataframe
that the rest of the project (EDA, recommender, evaluation) can rely on.

Standardized columns produced (when available in the source data):
    user_id, product_id, interaction, timestamp, product_name,
    category, brand, price, interaction_weight
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Interaction -> numeric weight used by the collaborative filtering model.
# Any interaction label not found here defaults to a weight of 1.
DEFAULT_INTERACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "add_to_cart": 3,
    "cart": 3,
    "purchase": 5,
    "buy": 5,
    "order": 5,
}


def _standardize_interaction_label(value: str) -> str:
    """Lowercase and normalize an interaction label for consistent weighting."""
    if not isinstance(value, str):
        return "unknown"
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def clean_dataset(df: pd.DataFrame, column_map: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and standardize the raw dataframe.

    Args:
        df: raw dataframe as loaded from disk.
        column_map: mapping of standardized column names -> actual column
            names, as produced by data_loader._map_columns.
        verbose: whether to print a cleaning summary (useful when run as a script).

    Returns:
        A new, cleaned dataframe with standardized column names.
    """
    original_shape = df.shape
    missing_before = df.isnull().sum().sum()

    work = df.copy()

    # --- Build the standardized dataframe ------------------------------
    clean = pd.DataFrame()
    clean["user_id"] = work[column_map["user_id"]].astype(str).str.strip()
    clean["product_id"] = work[column_map["product_id"]].astype(str).str.strip()
    clean["interaction"] = work[column_map["interaction"]].apply(_standardize_interaction_label)

    if "timestamp" in column_map:
        clean["timestamp"] = pd.to_datetime(work[column_map["timestamp"]], errors="coerce")
    else:
        clean["timestamp"] = pd.NaT

    if "product_name" in column_map:
        clean["product_name"] = work[column_map["product_name"]].astype(str).str.strip()
    else:
        clean["product_name"] = None

    if "category" in column_map:
        clean["category"] = work[column_map["category"]].astype(str).str.strip()
    else:
        clean["category"] = None

    if "brand" in column_map:
        clean["brand"] = work[column_map["brand"]].astype(str).str.strip()
    else:
        clean["brand"] = None

    if "price" in column_map:
        clean["price"] = pd.to_numeric(work[column_map["price"]], errors="coerce")
    else:
        clean["price"] = None

    # --- Validity checks -------------------------------------------------
    invalid_user = clean["user_id"].isin(["", "nan", "none", "None", "NaN"])
    invalid_product = clean["product_id"].isin(["", "nan", "none", "None", "NaN"])
    before_valid = len(clean)
    clean = clean[~(invalid_user | invalid_product)].copy()
    dropped_invalid = before_valid - len(clean)

    # --- Duplicate detection & removal -----------------------------------
    duplicate_count = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)

    # --- Interaction weights ----------------------------------------------
    clean["interaction_weight"] = clean["interaction"].map(DEFAULT_INTERACTION_WEIGHTS).fillna(1).astype(int)

    if verbose:
        print("=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Original shape:          {original_shape}")
        print(f"Cleaned shape:            {clean.shape}")
        print(f"Total missing (raw):      {missing_before}")
        print(f"Invalid user/product rows dropped: {dropped_invalid}")
        print(f"Duplicate records found & removed: {duplicate_count}")
        print(f"Unique users:             {clean['user_id'].nunique()}")
        print(f"Unique products:          {clean['product_id'].nunique()}")
        print(f"Total interactions:       {len(clean)}")
        print("Interaction type counts:")
        print(clean["interaction"].value_counts().to_string())
        print("=" * 60)

    return clean


def get_dataset_stats(clean_df: pd.DataFrame) -> dict:
    """Return a small dict of summary statistics used across the app/EDA/report."""
    return {
        "n_rows": len(clean_df),
        "n_users": clean_df["user_id"].nunique(),
        "n_products": clean_df["product_id"].nunique(),
        "n_interactions": len(clean_df),
        "interaction_counts": clean_df["interaction"].value_counts().to_dict(),
        "has_category": clean_df["category"].notna().any(),
        "has_brand": clean_df["brand"].notna().any(),
        "has_price": clean_df["price"].notna().any(),
        "has_timestamp": clean_df["timestamp"].notna().any(),
    }


if __name__ == "__main__":
    from data_loader import load_dataset

    df, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
    else:
        cleaned = clean_dataset(df, colmap)
        print(get_dataset_stats(cleaned))
