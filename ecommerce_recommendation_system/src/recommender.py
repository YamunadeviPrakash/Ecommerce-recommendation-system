"""
recommender.py
---------------
Item-based collaborative filtering recommendation engine.

Approach:
    1. Build a User x Item interaction matrix using weighted interaction
       scores (view=1, click=2, add_to_cart=3, purchase=5 by default).
    2. Compute Item-Item cosine similarity from that matrix (transposed).
    3. For a target user, look at the items they've already interacted with,
       and recommend the most similar *other* items, weighted by how
       strongly the user interacted with each of their known items.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedRecommender:
    """Item-based collaborative filtering recommender built from interaction data."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: cleaned interaction dataframe with at least the columns
                user_id, product_id, interaction_weight (see preprocessing.py).
        """
        self.df = df
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.index_user: list[str] = []
        self.index_item: list[str] = []
        self.user_item_matrix: Optional[sparse.csr_matrix] = None
        self.item_similarity: Optional[np.ndarray] = None
        self._product_lookup: Optional[pd.DataFrame] = None
        self._fit()

    # ------------------------------------------------------------------ #
    def _fit(self) -> None:
        """Build the user-item matrix and item-item similarity matrix."""
        agg = (
            self.df.groupby(["user_id", "product_id"])["interaction_weight"]
            .sum()
            .reset_index()
        )

        users = sorted(agg["user_id"].unique())
        items = sorted(agg["product_id"].unique())
        self.index_user = users
        self.index_item = items
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {p: i for i, p in enumerate(items)}

        rows = agg["user_id"].map(self.user_index)
        cols = agg["product_id"].map(self.item_index)
        vals = agg["interaction_weight"].astype(float)

        self.user_item_matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(users), len(items))
        )

        # Item-item cosine similarity (items are columns -> transpose to rows)
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)

        # Lookup table for product metadata (name/category/brand), first non-null wins
        meta_cols = [c for c in ["product_name", "category", "brand"] if c in self.df.columns]
        if meta_cols:
            self._product_lookup = (
                self.df[["product_id"] + meta_cols]
                .drop_duplicates(subset="product_id")
                .set_index("product_id")
            )
        else:
            self._product_lookup = pd.DataFrame(index=[]).reindex(columns=meta_cols)

    # ------------------------------------------------------------------ #
    def _product_meta(self, product_id: str) -> dict:
        meta = {"product_name": "N/A", "category": "N/A", "brand": "N/A"}
        if self._product_lookup is not None and product_id in self._product_lookup.index:
            row = self._product_lookup.loc[product_id]
            if isinstance(row, pd.DataFrame):  # duplicate safety
                row = row.iloc[0]
            for col in ["product_name", "category", "brand"]:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and str(val).strip().lower() not in ("", "none", "nan"):
                        meta[col] = val
        return meta

    def known_users(self) -> list[str]:
        return self.index_user

    def user_history(self, user_id: str) -> pd.DataFrame:
        """Return this user's historical interactions, most recent first if timestamp exists."""
        hist = self.df[self.df["user_id"] == user_id].copy()
        if "timestamp" in hist.columns and hist["timestamp"].notna().any():
            hist = hist.sort_values("timestamp", ascending=False)
        return hist

    # ------------------------------------------------------------------ #
    def get_recommendations(
        self,
        user_id: str,
        n: int = 5,
        category_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate top-N item recommendations for a user.

        Returns a DataFrame with columns:
            rank, product_id, product_name, category, score
        (empty DataFrame if the user is unknown or has no viable recommendations)
        """
        columns = ["rank", "product_id", "product_name", "category", "score"]

        if user_id not in self.user_index:
            return pd.DataFrame(columns=columns)

        u_idx = self.user_index[user_id]
        user_row = self.user_item_matrix[u_idx].toarray().ravel()  # weights per item
        interacted_mask = user_row > 0

        if not interacted_mask.any():
            return pd.DataFrame(columns=columns)

        # Score for each candidate item = weighted sum of similarity to items the
        # user has already interacted with (weighted by how strongly they engaged).
        scores = self.item_similarity[:, interacted_mask] @ user_row[interacted_mask]

        # Exclude items already interacted with
        scores[interacted_mask] = -np.inf

        candidate_order = np.argsort(scores)[::-1]

        results = []
        for idx in candidate_order:
            if len(results) >= n:
                break
            score = scores[idx]
            if not np.isfinite(score) or score <= 0:
                continue
            product_id = self.index_item[idx]
            meta = self._product_meta(product_id)

            if category_filter and category_filter != "All":
                if str(meta.get("category")) != str(category_filter):
                    continue

            results.append(
                {
                    "product_id": product_id,
                    "product_name": meta["product_name"],
                    "category": meta["category"],
                    "score": round(float(score), 4),
                }
            )

        if not results:
            return pd.DataFrame(columns=columns)

        out = pd.DataFrame(results)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out[columns]


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import clean_dataset

    raw, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
    else:
        cleaned = clean_dataset(raw, colmap, verbose=False)
        rec = ItemBasedRecommender(cleaned)
        sample_user = rec.known_users()[0]
        print(f"Sample recommendations for user: {sample_user}")
        print(rec.get_recommendations(sample_user, n=5).to_string(index=False))
