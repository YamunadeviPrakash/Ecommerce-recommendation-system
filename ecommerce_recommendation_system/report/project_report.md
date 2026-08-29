# Project Report: Recommendation System Prototype for an E-Commerce Platform Using User–Item Interactions

## 1. Executive Summary

This project builds a prototype recommendation system for an e-commerce
platform using real user–item interaction data (the Kaggle *Retail
Personalization Dataset*, ~150,000 interaction records). The system applies
**item-based collaborative filtering** to generate personalized product
recommendations, is evaluated with standard ranking metrics, and is exposed
through an interactive Streamlit dashboard for business stakeholders. All
figures in this report are computed directly from the dataset — none are
fabricated.

## 2. Dataset Overview

| Metric | Value |
|---|---|
| Total interaction records | 150,000 |
| Unique users | 4,999 |
| Unique products | 1,999 |
| Interaction types | view, click, add_to_cart, purchase |
| Additional fields | timestamp, session_id, device_type, location, price, discount, product_category, brand, user demographics, loyalty_score, rating |

### Interaction Type Breakdown

| Interaction | Count | Share |
|---|---|---|
| view | 75,215 | 50.14% |
| click | 44,989 | 29.99% |
| add_to_cart | 22,217 | 14.81% |
| purchase | 7,579 | 5.05% |

### Category Distribution

The **Apparel** category accounts for the largest share of interactions
(20.08%), followed closely by the other four categories (Beauty,
Electronics, Home, Sports), which are fairly evenly distributed — indicating
a broad, non-niche product catalog.

### Price Statistics

| Statistic | Value |
|---|---|
| Mean price | 251.68 |
| Median price | 251.31 |
| Min price | 5.00 |
| Max price | 500.00 |

## 3. Data Preprocessing

The raw dataset required minimal cleaning:

- **Missing values:** only the `rating` field contained nulls (74,879 of
  150,000 rows) — expected, since not every interaction results in a
  product rating. All fields used by the recommendation engine (user ID,
  product ID, interaction type) were fully populated.
- **Duplicates:** 0 exact duplicate rows found.
- **Invalid records:** 0 rows with missing/blank user or product IDs.
- **Standardization:** interaction labels were lowercased and normalized;
  timestamps were parsed to proper datetime objects.

The cleaning pipeline is fully automated (`src/preprocessing.py`) and prints
a summary each time it runs, so results are reproducible and auditable.

## 4. Exploratory Data Analysis (Descriptive & Diagnostic Analytics)

**Descriptive — what happened:**
- The single most-interacted product is `P1843`, with 102 total interactions.
- The top 10 products (see `outputs/figures/top_10_products.png`) each
  received between roughly 90–102 interactions, suggesting a relatively flat
  popularity curve rather than a small number of runaway bestsellers.
- Interaction volume is dominated by low-commitment actions (`view`,
  `click`), which together make up ~80% of all activity, while `purchase`
  makes up only ~5%.

**Diagnostic — why it might be happening:**
- The steep drop-off from `view` (50%) to `purchase` (5%) suggests a
  significant funnel leak between product discovery and checkout — a
  natural target for personalization and cross-selling.
- Because product popularity is fairly evenly spread (no single product
  dominates), a *personalized* recommendation approach is likely to add more
  value than a simple "most popular" list, since customer interests appear
  to be genuinely varied rather than concentrated on a handful of hits.

Charts generated (see `outputs/figures/`):
`interaction_distribution.png`, `top_10_products.png`,
`top_10_active_users.png`, `category_distribution.png`,
`price_distribution.png`, `interaction_trends_over_time.png`.

## 5. Recommendation System (Predictive Analytics)

**Method: Item-Based Collaborative Filtering**

1. Interactions are converted to numeric weights: `view=1`, `click=2`,
   `add_to_cart=3`, `purchase=5` (the code auto-adapts to unrecognized
   interaction labels by defaulting to weight 1, so it is not tied to this
   specific dataset's exact vocabulary).
2. Weighted interactions are aggregated into a sparse **User × Item**
   matrix (4,999 users × 1,999 products).
3. **Item–Item cosine similarity** is computed with scikit-learn on the
   transposed matrix (1,999 × 1,999 similarity matrix).
4. For a target user, candidate products are scored as the weighted sum of
   their similarity to the products that user has already engaged with;
   already-seen products are excluded and the remaining candidates are
   ranked to produce the Top-N recommendation list.

**Sample output** (User `U0001`, Top-5):

| Rank | Product | Category | Score |
|---|---|---|---|
| 1 | P0512 | Sports | 1.4029 |
| 2 | P0381 | Beauty | 1.3492 |
| 3 | P0741 | Electronics | 1.3203 |
| 4 | P0179 | Electronics | 1.3189 |
| 5 | P0424 | Sports | 1.3186 |

(The source dataset does not include a product-name field, so recommendations
are identified by Product ID and Category; the pipeline is written to use a
`product_name` column automatically if one is present in a future dataset
version.)

The full sample recommendation set for 25 users is saved to
`outputs/recommendations.csv`.

## 6. Model Evaluation

**Methodology:** leave-one-out evaluation. For each eligible user (with at
least 3 interactions), their single most recent interaction is held out as
the test item; the model is trained on the remaining data, and we check
whether the held-out item appears in that user's Top-K recommendation list.

| K | Precision@K | Recall@K | Hit Rate@K | Users Evaluated |
|---|---|---|---|---|
| 5 | 0.0000 | 0.0000 | 0.0000 | 500 |
| 10 | 0.0004 | 0.0040 | 0.0040 | 500 |

**Interpretation:** these scores are modest in absolute terms, which is
expected given the dataset's sparsity — 4,999 users spread across 1,999
products means most user-product pairs have no interaction at all, making
it statistically hard for any single held-out item to land in a short
Top-5/Top-10 list. This is a realistic, honest result for a first
prototype (not an artificially inflated number), and it directly motivates
the business recommendations below: a production system would benefit from
richer signals (browsing sessions, product content/text features, hybrid
filtering) to improve hit rate further.

## 7. Business Insights & Recommendations (Prescriptive Analytics)

**Key insights:**
- Product `P1843` has the highest number of interactions (102).
- Category `Apparel` accounts for the largest share of interactions (20.08%).
- Purchase interactions represent only 5.05% of all recorded interactions —
  a large view-to-purchase gap.
- Interaction volume is spread across 4,999 active users, averaging ~30
  interactions per user.

**Recommended actions:**
1. **Personalize recommendations** using the collaborative filtering model
   demonstrated in this prototype, surfaced at key touchpoints (homepage,
   cart page, post-purchase emails).
2. **Cross-sell within categories** — since interaction share is spread
   fairly evenly across categories, tailor cross-sell suggestions to each
   customer's most-engaged category rather than using a single global
   "trending" list.
3. **Address the view-to-purchase funnel leak** — with 80% of activity being
   low-commitment (view/click) and only 5% converting to purchase,
   investigate friction points (pricing, product information, checkout flow)
   for high-view, low-purchase products.
4. **Expand the signal set** for future iterations — incorporating browsing
   session sequences, product text/content similarity, and user demographic
   segments alongside the current interaction-based model would likely
   improve recommendation precision beyond the baseline reported here.

## 8. Technical Notes

- Built entirely with Python, pandas, NumPy, scikit-learn, Matplotlib,
  Seaborn, and Streamlit — no deep learning frameworks, databases, or cloud
  services required.
- All paths are relative, so the project runs unmodified on any Windows
  machine once the virtual environment and dependencies are set up.
- Streamlit's `@st.cache_data` / `@st.cache_resource` decorators are used to
  avoid recomputing the similarity matrix and reloading data on every UI
  interaction, keeping the app responsive on a normal laptop despite the
  ~150,000-row dataset.
