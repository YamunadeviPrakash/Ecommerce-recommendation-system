"""
evaluation.py
--------------
Simple train/test evaluation of the item-based recommender.

Methodology:
    For each user with enough history, hold out their single most recent
    interaction (the "test item") and train the recommender on everything
    else. Then check whether the held-out item appears in the user's
    top-N recommendation list generated from their remaining history.

Metrics (averaged across evaluated users):
    Precision@K = (# relevant items in top-K) / K
                  (with 1 relevant item per user, this is 1/K if hit else 0)
    Recall@K    = (# relevant items in top-K) / (# relevant items held out)
                  (with exactly 1 held-out item per user, this is 1 if hit else 0)
    Hit Rate@K  = fraction of users for whom the held-out item appears in top-K
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender import ItemBasedRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def train_test_split_leave_one_out(df: pd.DataFrame, min_interactions: int = 3, max_eval_users: int = 500):
    """
    For each eligible user (>= min_interactions interactions), hold out one
    interaction as the test item. Returns (train_df, test_dict) where
    test_dict maps user_id -> held-out product_id.

    max_eval_users caps the number of users evaluated, for performance on
    large datasets.
    """
    counts = df.groupby("user_id").size()
    eligible_users = counts[counts >= min_interactions].index.tolist()

    if len(eligible_users) > max_eval_users:
        # Deterministic sample for reproducibility
        eligible_users = sorted(eligible_users)[:max_eval_users]

    test_dict = {}
    held_out_indices = []

    has_time = "timestamp" in df.columns and df["timestamp"].notna().any()

    for user in eligible_users:
        user_rows = df[df["user_id"] == user]
        if has_time and user_rows["timestamp"].notna().any():
            test_row = user_rows.sort_values("timestamp", ascending=False).iloc[0]
        else:
            test_row = user_rows.iloc[-1]
        test_dict[user] = test_row["product_id"]
        held_out_indices.append(test_row.name)

    train_df = df.drop(index=held_out_indices).reset_index(drop=True)
    return train_df, test_dict


def evaluate_at_k(df: pd.DataFrame, k: int = 5, min_interactions: int = 3, max_eval_users: int = 500) -> dict:
    """
    Run the leave-one-out evaluation for a given K.

    Returns a dict with precision, recall, hit_rate, and n_users_evaluated.
    """
    train_df, test_dict = train_test_split_leave_one_out(df, min_interactions, max_eval_users)

    if not test_dict:
        return {"k": k, "precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "n_users_evaluated": 0}

    recommender = ItemBasedRecommender(train_df)

    hits = 0
    precisions = []
    recalls = []
    evaluated = 0

    for user, held_out_item in test_dict.items():
        if user not in recommender.user_index:
            continue
        recs = recommender.get_recommendations(user, n=k)
        evaluated += 1
        if recs.empty:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        hit = held_out_item in recs["product_id"].values
        if hit:
            hits += 1
        precisions.append((1 if hit else 0) / k)
        recalls.append(1 if hit else 0)  # exactly 1 relevant item per user

    if evaluated == 0:
        return {"k": k, "precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "n_users_evaluated": 0}

    return {
        "k": k,
        "precision": round(sum(precisions) / evaluated, 4),
        "recall": round(sum(recalls) / evaluated, 4),
        "hit_rate": round(hits / evaluated, 4),
        "n_users_evaluated": evaluated,
    }


def run_evaluation(df: pd.DataFrame, ks=(5, 10), min_interactions: int = 3, max_eval_users: int = 500) -> pd.DataFrame:
    """Run evaluation for each K in `ks` and return results as a DataFrame."""
    rows = [evaluate_at_k(df, k=k, min_interactions=min_interactions, max_eval_users=max_eval_users) for k in ks]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import clean_dataset

    raw, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
    else:
        cleaned = clean_dataset(raw, colmap, verbose=False)
        results = run_evaluation(cleaned, ks=(5, 10))
        print("Evaluation results:")
        print(results.to_string(index=False))

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUTS_DIR / "evaluation_results.csv", index=False)
        print(f"\nSaved to {OUTPUTS_DIR / 'evaluation_results.csv'}")
