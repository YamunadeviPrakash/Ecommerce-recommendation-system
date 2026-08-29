"""
eda.py
------
Business-analytics oriented exploratory data analysis.

Produces summary tables (as DataFrames, easy to reuse in Streamlit) and
saves a standard set of charts to outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless/script execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def top_n_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    counts = df["product_id"].value_counts().head(n).reset_index()
    counts.columns = ["product_id", "interaction_count"]
    return counts


def top_n_users(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    counts = df["user_id"].value_counts().head(n).reset_index()
    counts.columns = ["user_id", "interaction_count"]
    return counts


def interaction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["interaction"].value_counts().reset_index()
    counts.columns = ["interaction", "count"]
    return counts


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df["category"].isna().all():
        return pd.DataFrame(columns=["category", "count"])
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    return counts


def brand_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df["brand"].isna().all():
        return pd.DataFrame(columns=["brand", "count"])
    counts = df["brand"].value_counts().reset_index()
    counts.columns = ["brand", "count"]
    return counts


def price_summary(df: pd.DataFrame) -> dict:
    if df["price"].isna().all():
        return {}
    return {
        "mean": float(df["price"].mean()),
        "median": float(df["price"].median()),
        "min": float(df["price"].min()),
        "max": float(df["price"].max()),
        "std": float(df["price"].std()),
    }


def interactions_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df["timestamp"].isna().all():
        return pd.DataFrame(columns=["period", "count"])
    ts = df.dropna(subset=["timestamp"]).copy()
    ts["period"] = ts["timestamp"].dt.to_period(freq).dt.to_timestamp()
    trend = ts.groupby("period").size().reset_index(name="count")
    return trend


def purchase_rate(df: pd.DataFrame) -> float:
    total = len(df)
    if total == 0:
        return 0.0
    purchases = (df["interaction"] == "purchase").sum()
    return round(100 * purchases / total, 2)


def run_full_eda(df: pd.DataFrame, save_figures: bool = True) -> dict:
    """
    Run the full EDA suite, optionally saving charts to outputs/figures/,
    and return a dict of the computed summary tables/values for reuse
    (e.g. by the Streamlit app or the analysis_summary.txt writer).
    """
    if save_figures:
        _ensure_dir(FIGURES_DIR)

    results = {
        "n_users": df["user_id"].nunique(),
        "n_products": df["product_id"].nunique(),
        "n_interactions": len(df),
        "interaction_dist": interaction_distribution(df),
        "top_products": top_n_products(df, 10),
        "top_users": top_n_users(df, 10),
        "category_dist": category_distribution(df),
        "brand_dist": brand_distribution(df),
        "price_stats": price_summary(df),
        "time_trend": interactions_over_time(df),
        "purchase_rate_pct": purchase_rate(df),
    }

    if save_figures:
        _plot_interaction_distribution(results["interaction_dist"])
        _plot_top_products(results["top_products"])
        _plot_category_distribution(results["category_dist"])
        _plot_user_activity(results["top_users"])
        if results["price_stats"]:
            _plot_price_distribution(df)
        if not results["time_trend"].empty:
            _plot_time_trend(results["time_trend"])

    return results


# ------------------------------------------------------------------ #
# Chart helpers
# ------------------------------------------------------------------ #

def _plot_interaction_distribution(dist: pd.DataFrame) -> None:
    if dist.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=dist, x="interaction", y="count", hue="interaction", palette="viridis", ax=ax, legend=False)
    ax.set_title("Interaction Type Distribution")
    ax.set_xlabel("Interaction Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "interaction_distribution.png", dpi=150)
    plt.close(fig)


def _plot_top_products(top_products: pd.DataFrame) -> None:
    if top_products.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_products.sort_values("interaction_count"),
        x="interaction_count", y="product_id", hue="product_id", palette="mako", ax=ax, legend=False,
    )
    ax.set_title("Top 10 Products by Interaction Count")
    ax.set_xlabel("Interaction Count")
    ax.set_ylabel("Product ID")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "top_10_products.png", dpi=150)
    plt.close(fig)


def _plot_category_distribution(cat_dist: pd.DataFrame) -> None:
    if cat_dist.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=cat_dist, x="count", y="category", hue="category", palette="crest", ax=ax, legend=False)
    ax.set_title("Category Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "category_distribution.png", dpi=150)
    plt.close(fig)


def _plot_user_activity(top_users: pd.DataFrame) -> None:
    if top_users.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_users.sort_values("interaction_count"),
        x="interaction_count", y="user_id", hue="user_id", palette="flare", ax=ax, legend=False,
    )
    ax.set_title("Top 10 Most Active Users")
    ax.set_xlabel("Interaction Count")
    ax.set_ylabel("User ID")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "top_10_active_users.png", dpi=150)
    plt.close(fig)


def _plot_price_distribution(df: pd.DataFrame) -> None:
    prices = df["price"].dropna()
    if prices.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(prices, bins=40, kde=True, color="teal", ax=ax)
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "price_distribution.png", dpi=150)
    plt.close(fig)


def _plot_time_trend(trend: pd.DataFrame) -> None:
    if trend.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trend["period"], trend["count"], color="darkorange", linewidth=1.5)
    ax.set_title("Interaction Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Interactions")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "interaction_trends_over_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import clean_dataset

    raw, colmap, err = load_dataset()
    if err:
        print(f"[ERROR] {err}")
    else:
        cleaned = clean_dataset(raw, colmap)
        res = run_full_eda(cleaned, save_figures=True)
        print(f"Saved figures to: {FIGURES_DIR}")
        print(f"Purchase rate: {res['purchase_rate_pct']}%")
