"""
run_pipeline.py
----------------
Runs the full offline pipeline (load -> clean -> EDA -> recommend -> evaluate)
and writes all output artifacts to outputs/:

    outputs/figures/*.png
    outputs/recommendations.csv
    outputs/evaluation_results.csv
    outputs/analysis_summary.txt

This is useful for regenerating outputs without launching the Streamlit app,
and is what the project report's numbers are drawn from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_dataset
from preprocessing import clean_dataset, get_dataset_stats
from eda import run_full_eda
from recommender import ItemBasedRecommender
from evaluation import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def generate_sample_recommendations(df: pd.DataFrame, recommender: ItemBasedRecommender, n_users: int = 25, n_recs: int = 5) -> pd.DataFrame:
    """Generate recommendations for a sample of users and combine into one table."""
    sample_users = recommender.known_users()[:n_users]
    frames = []
    for user in sample_users:
        recs = recommender.get_recommendations(user, n=n_recs)
        if not recs.empty:
            recs = recs.copy()
            recs.insert(0, "user_id", user)
            frames.append(recs)
    if not frames:
        return pd.DataFrame(columns=["user_id", "rank", "product_id", "product_name", "category", "score"])
    return pd.concat(frames, ignore_index=True)


def write_analysis_summary(df: pd.DataFrame, stats: dict, eda_results: dict, eval_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("E-COMMERCE RECOMMENDATION SYSTEM - ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total interactions analyzed: {stats['n_interactions']:,}")
    lines.append(f"Unique users:                {stats['n_users']:,}")
    lines.append(f"Unique products:             {stats['n_products']:,}")
    lines.append("")

    lines.append("-- Interaction Type Breakdown --")
    total = stats["n_interactions"]
    for interaction, count in stats["interaction_counts"].items():
        pct = 100 * count / total if total else 0
        lines.append(f"  {interaction:15s}: {count:>8,}  ({pct:5.2f}%)")
    lines.append("")

    most_interacted_product = df["product_id"].value_counts().idxmax()
    lines.append(f"Product '{most_interacted_product}' has the highest number of interactions "
                 f"({df['product_id'].value_counts().max():,}).")

    if df["category"].notna().any():
        top_cat = df["category"].value_counts().idxmax()
        top_cat_pct = 100 * df["category"].value_counts().max() / total
        lines.append(f"Category '{top_cat}' accounts for the largest share of interactions "
                     f"({top_cat_pct:.2f}%).")

    if "purchase" in stats["interaction_counts"]:
        purchase_pct = 100 * stats["interaction_counts"]["purchase"] / total
        lines.append(f"Purchase interactions represent {purchase_pct:.2f}% of all recorded interactions.")

    if eda_results.get("price_stats"):
        p = eda_results["price_stats"]
        lines.append("")
        lines.append("-- Price Statistics --")
        lines.append(f"  Mean price:   {p['mean']:.2f}")
        lines.append(f"  Median price: {p['median']:.2f}")
        lines.append(f"  Min / Max:    {p['min']:.2f} / {p['max']:.2f}")

    lines.append("")
    lines.append("-- Top 5 Products by Interaction Count --")
    for _, row in eda_results["top_products"].head(5).iterrows():
        lines.append(f"  {row['product_id']:10s} : {row['interaction_count']:,} interactions")

    lines.append("")
    lines.append("-- Model Evaluation (Leave-One-Out) --")
    for _, row in eval_df.iterrows():
        lines.append(
            f"  K={int(row['k']):2d}  Precision@K={row['precision']:.4f}  "
            f"Recall@K={row['recall']:.4f}  HitRate@K={row['hit_rate']:.4f}  "
            f"(evaluated on {int(row['n_users_evaluated'])} users)"
        )

    lines.append("")
    lines.append("-- Business Recommendations --")
    lines.append("  1. Personalize product recommendations per customer using the collaborative filtering model.")
    lines.append("  2. Cross-sell related products within a customer's most-engaged categories.")
    lines.append("  3. Prioritize marketing/promotions around top-performing products and categories.")
    lines.append("  4. Investigate high-view / low-purchase products to improve conversion.")
    lines.append("")
    lines.append("=" * 70)

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading dataset...")
    raw, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
        return

    print("Cleaning dataset...")
    df = clean_dataset(raw, colmap, verbose=True)
    stats = get_dataset_stats(df)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRunning EDA and saving figures...")
    eda_results = run_full_eda(df, save_figures=True)

    print("\nTraining recommendation model...")
    recommender = ItemBasedRecommender(df)

    print("Generating sample recommendations...")
    sample_recs = generate_sample_recommendations(df, recommender, n_users=25, n_recs=5)
    sample_recs.to_csv(OUTPUTS_DIR / "recommendations.csv", index=False)
    print(f"  Saved: {OUTPUTS_DIR / 'recommendations.csv'} ({len(sample_recs)} rows)")

    print("\nRunning model evaluation...")
    eval_df = run_evaluation(df, ks=(5, 10))
    eval_df.to_csv(OUTPUTS_DIR / "evaluation_results.csv", index=False)
    print(eval_df.to_string(index=False))
    print(f"  Saved: {OUTPUTS_DIR / 'evaluation_results.csv'}")

    print("\nWriting analysis summary...")
    write_analysis_summary(df, stats, eda_results, eval_df, OUTPUTS_DIR / "analysis_summary.txt")
    print(f"  Saved: {OUTPUTS_DIR / 'analysis_summary.txt'}")

    print("\nPipeline complete. All outputs written to:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
