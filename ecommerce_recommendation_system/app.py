"""
app.py
------
Streamlit web application for the E-Commerce Recommendation System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.data_loader import load_dataset  # noqa: E402
from src.preprocessing import clean_dataset, get_dataset_stats  # noqa: E402
from src.eda import (  # noqa: E402
    interaction_distribution,
    top_n_products,
    category_distribution,
    top_n_users,
)
from src.recommender import ItemBasedRecommender  # noqa: E402

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛍️",
    layout="wide",
)


# ---------------------------------------------------------------------- #
# Cached loading / heavy computation
# ---------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def cached_load_and_clean():
    raw, colmap, err = load_dataset()
    if err:
        return None, err
    cleaned = clean_dataset(raw, colmap, verbose=False)
    return cleaned, None


@st.cache_resource(show_spinner="Training recommendation model...")
def cached_recommender(df: pd.DataFrame):
    return ItemBasedRecommender(df)


# ---------------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------------- #
st.title("🛍️ E-Commerce Recommendation System")
st.markdown("**Personalized Product Recommendations Using User–Item Interactions**")
st.divider()

df, error = cached_load_and_clean()

if error:
    st.error(f"Dataset not found. Please place the downloaded CSV file inside the data folder.\n\nDetails: {error}")
    st.stop()

if df is None or df.empty or df["user_id"].nunique() == 0:
    st.warning("No users are available in the dataset. Please check the data file.")
    st.stop()

recommender = cached_recommender(df)
stats = get_dataset_stats(df)

# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
st.sidebar.header("Controls")

user_list = recommender.known_users()
selected_user = st.sidebar.selectbox("Select Customer", options=user_list, index=0)

n_recs = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=10, value=5)

available_categories = ["All"]
if df["category"].notna().any():
    available_categories += sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category Filter (optional)", options=available_categories, index=0)

st.sidebar.divider()
st.sidebar.caption(
    "Business Analytics Prototype — item-based collaborative filtering "
    "over real user–item interaction data."
)

# ---------------------------------------------------------------------- #
# KPI Dashboard
# ---------------------------------------------------------------------- #
most_popular_product = df["product_id"].value_counts().idxmax()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Users", f"{stats['n_users']:,}")
kpi2.metric("Total Products", f"{stats['n_products']:,}")
kpi3.metric("Total Interactions", f"{stats['n_interactions']:,}")
kpi4.metric("Most Popular Product", most_popular_product)

st.divider()

# ---------------------------------------------------------------------- #
# Section 1 — Customer Profile
# ---------------------------------------------------------------------- #
st.header("👤 Customer Profile")

history = recommender.user_history(selected_user)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("Selected User", selected_user)
    st.metric("Number of Interactions", len(history))
    if df["product_id"].nunique() > 0:
        st.metric("Distinct Products Interacted With", history["product_id"].nunique())

with col_b:
    st.subheader("Recent Interactions")
    display_cols = [c for c in ["product_id", "product_name", "interaction", "category", "timestamp"] if c in history.columns]
    rename_map = {
        "product_id": "Product",
        "product_name": "Product Name",
        "interaction": "Interaction",
        "category": "Category",
        "timestamp": "Timestamp",
    }
    if history.empty:
        st.info("This user has no recorded interactions.")
    else:
        preview = history[display_cols].head(10).rename(columns=rename_map)
        st.dataframe(preview, hide_index=True)

st.divider()

# ---------------------------------------------------------------------- #
# Section 2 — Personalized Recommendations
# ---------------------------------------------------------------------- #
st.header("✨ Recommended Products for You")

recs = recommender.get_recommendations(selected_user, n=n_recs, category_filter=selected_category)

if recs.empty:
    st.info("No recommendations available for this user.")
else:
    display_recs = recs.rename(
        columns={
            "rank": "Rank",
            "product_id": "Product",
            "product_name": "Product Name",
            "category": "Category",
            "score": "Recommendation Score",
        }
    )

    def _highlight_top(row):
        if row["Rank"] == 1:
            return ["background-color: #fff3b0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_recs.style.apply(_highlight_top, axis=1),
        hide_index=True
    )

    st.caption(
        "These products are recommended based on similarities between products "
        "previously interacted with by this customer and products interacted "
        "with by other customers."
    )

st.divider()

# ---------------------------------------------------------------------- #
# Section 3 — Business Analytics
# ---------------------------------------------------------------------- #
st.header("📊 Business Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Interaction Types", "Top 10 Products", "Category Distribution", "User Activity"])

with tab1:
    dist = interaction_distribution(df)
    st.bar_chart(dist.set_index("interaction"))

with tab2:
    top_products = top_n_products(df, 10)
    st.bar_chart(top_products.set_index("product_id"))

with tab3:
    cat_dist = category_distribution(df)
    if cat_dist.empty:
        st.info("Category information is not available in this dataset.")
    else:
        st.bar_chart(cat_dist.set_index("category"))

with tab4:
    top_users = top_n_users(df, 10)
    st.bar_chart(top_users.set_index("user_id"))

st.divider()

# ---------------------------------------------------------------------- #
# Section 4 — Business Insights
# ---------------------------------------------------------------------- #
st.header("💡 Business Insights")

interaction_counts = stats["interaction_counts"]
total_interactions = stats["n_interactions"]
most_common_interaction = max(interaction_counts, key=interaction_counts.get) if interaction_counts else "N/A"
most_interacted_product = df["product_id"].value_counts().idxmax()

insights = [
    f"**Most interacted product:** `{most_interacted_product}` "
    f"({df['product_id'].value_counts().max():,} interactions).",
    f"**Most common interaction type:** `{most_common_interaction}` "
    f"({interaction_counts.get(most_common_interaction, 0):,} occurrences, "
    f"{100 * interaction_counts.get(most_common_interaction, 0) / total_interactions:.1f}% of all interactions).",
    f"**Number of active users:** {stats['n_users']:,} users generated {total_interactions:,} total interactions "
    f"(avg. {total_interactions / stats['n_users']:.1f} interactions per user).",
]

if df["category"].notna().any():
    top_cat = df["category"].value_counts().idxmax()
    top_cat_pct = 100 * df["category"].value_counts().max() / total_interactions
    insights.append(f"**Most popular category:** `{top_cat}` accounts for {top_cat_pct:.1f}% of all interactions.")

if "purchase" in interaction_counts:
    purchase_pct = 100 * interaction_counts["purchase"] / total_interactions
    insights.append(f"**Purchase conversion:** Purchase interactions represent {purchase_pct:.1f}% of all recorded interactions.")

for insight in insights:
    st.markdown(f"- {insight}")

with st.expander("Business Analytics Framework"):
    st.markdown(
        """
**Descriptive Analytics** — *What happened?*
Popular products, active customers, and interaction patterns shown above summarize past behavior.

**Diagnostic Analytics** — *Why might it have happened?*
Category and interaction-type breakdowns help explain which product types and touchpoints drive engagement.

**Predictive Analytics** — *What might the customer want?*
The item-based collaborative filtering model predicts products a customer is likely to engage with next,
based on similarity to their historical interactions.

**Prescriptive Analytics** — *What should the business do?*
- Personalize product recommendations per customer using the model above.
- Cross-sell related products within a customer's most-engaged categories.
- Prioritize engagement campaigns around top-performing products/categories.
- Investigate low-converting categories to improve the purchase funnel.
        """
    )

st.divider()
st.caption("Prototype Business Analytics project — item-based collaborative filtering on real interaction data.")
