# 🛍️ E-Commerce Recommendation System (Business Analytics Prototype)

**Recommendation System Prototype for an E-Commerce Platform Using User–Item Interactions**

A complete, beginner-friendly Business Analytics project that builds an
**item-based collaborative filtering** recommendation engine on real
e-commerce interaction data, wraps it in an interactive **Streamlit**
dashboard, and documents the analysis in a Jupyter notebook and written report.

This guide assumes **Windows + Visual Studio Code**.

---

## Dataset

This project uses the **Retail Personalization Dataset** (~150,000 e-commerce
interaction records):
https://www.kaggle.com/datasets/ziya07/retail-personalization-dataset

A copy of the dataset (`data/dataset.csv`) is already included in this
project folder, so it will work immediately with no extra downloads.

If you ever need to replace it with a fresh copy from Kaggle, just download
the CSV and drop it into the `data/` folder — the app automatically detects
any CSV/Excel file placed there (see **Step 7** below).

---

## Requirements

- **Python 3.10 or 3.11** — https://www.python.org/downloads/
  (during install, check **"Add Python to PATH"**)
- **Visual Studio Code** — https://code.visualstudio.com/
- **Python extension for VS Code** (install from the Extensions panel in VS Code)

---

## Step 1 — Extract the ZIP

Extract `ecommerce_recommendation_system.zip` to a location of your choice,
for example:

```
C:\Projects\ecommerce_recommendation_system
```

---

## Step 2 — Open the Project in VS Code

1. Open VS Code.
2. Go to **File → Open Folder…**
3. Select the extracted `ecommerce_recommendation_system` folder.

---

## Step 3 — Open a Terminal in VS Code

Go to **Terminal → New Terminal** (or press `` Ctrl+` ``). All commands below
are run from this terminal, inside the project folder.

---

## Step 4 — Create a Virtual Environment

```powershell
python -m venv venv
```

This creates an isolated Python environment in a `venv` folder so project
dependencies don't clash with anything else on your machine.

---

## Step 5 — Activate the Virtual Environment

**PowerShell** (VS Code's default terminal on Windows):

```powershell
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script with an execution-policy error, run this
> once, then try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**Command Prompt** (if you're using `cmd.exe` instead):

```cmd
venv\Scripts\activate
```

You'll know it worked when you see `(venv)` at the start of your terminal prompt.

---

## Step 6 — Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs pandas, numpy, matplotlib, seaborn, scikit-learn, scipy,
streamlit, and jupyter.

---

## Step 7 — Add / Verify the Dataset

The dataset should already be at:

```
data\dataset.csv
```

If it's missing (or you want to refresh it), download the CSV from the
Kaggle link above and place it inside the `data\` folder. The app
automatically detects `.csv` or `.xlsx` files placed there — no code changes
needed.

---

## Step 8 — Run the Streamlit App

```powershell
streamlit run app.py
```

Alternatively, just double-click **`run_app.bat`** in File Explorer — it will
create the virtual environment, install dependencies (only if needed), and
launch the app automatically.

---

## Step 9 — Open It in Your Browser

Streamlit will print a local URL in the terminal, typically:

```
Local URL: http://localhost:8501
```

It usually opens automatically in your default browser. If not, copy the URL
into your browser manually.

---

## Step 10 — Run the Jupyter Notebook (Optional, for the detailed analysis)

With the virtual environment still active, you can either:

**Option A — via VS Code:**
Open `notebooks/recommendation_analysis.ipynb` in VS Code, select the `venv`
Python kernel (top-right of the notebook), and run all cells.

**Option B — via terminal:**

```powershell
python -m notebook
```

This opens the classic Jupyter Notebook interface in your browser, where you
can open `notebooks/recommendation_analysis.ipynb`.

---

## Project Structure

```
ecommerce_recommendation_system/
│
├── app.py                     # Streamlit web application (main demo)
├── requirements.txt
├── README.md
├── run_app.bat                 # One-click Windows launcher
├── .gitignore
│
├── data/
│   └── dataset.csv             # Retail Personalization Dataset (~150k rows)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Locates & loads the dataset, maps columns
│   ├── preprocessing.py        # Cleaning, validation, standardization
│   ├── eda.py                  # Exploratory data analysis + chart generation
│   ├── recommender.py          # Item-based collaborative filtering engine
│   ├── evaluation.py           # Precision@K / Recall@K / Hit Rate@K
│   └── run_pipeline.py         # Regenerates all outputs/ artifacts in one go
│
├── notebooks/
│   └── recommendation_analysis.ipynb   # Full walkthrough analysis
│
├── outputs/
│   ├── figures/                 # Saved PNG charts
│   ├── recommendations.csv      # Sample recommendations for 25 users
│   ├── evaluation_results.csv   # Model evaluation metrics
│   └── analysis_summary.txt     # Auto-generated business insights
│
└── report/
    └── project_report.md        # Written business analytics report
```

---

## How the Recommendation Engine Works

1. **Interaction weighting** — each interaction type is converted to a
   numeric weight: `view=1`, `click=2`, `add_to_cart=3`, `purchase=5`
   (unrecognized interaction labels default to `1`, so the code adapts to
   datasets with different interaction naming).
2. **User × Item matrix** — interactions are aggregated into a sparse
   user-item matrix of weighted engagement scores.
3. **Item-Item Cosine Similarity** — computed with scikit-learn on the
   transposed matrix, capturing which products tend to be engaged with by
   the same users.
4. **Recommendation scoring** — for a chosen user, each candidate product's
   score is the weighted sum of its similarity to the products that user has
   already interacted with. Already-interacted products are excluded, and
   results are sorted to return the Top-N.

## Regenerating All Outputs

To rerun the whole offline pipeline (EDA charts, recommendations,
evaluation, and the summary text file) without opening Streamlit:

```powershell
cd src
python run_pipeline.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python and check "Add Python to PATH" |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| "Dataset not found" in the app | Make sure a CSV/Excel file is in the `data\` folder |
| Streamlit doesn't open a browser tab | Manually visit the Local URL shown in the terminal |
| `pip install` is slow/fails | Check your internet connection, then retry `pip install -r requirements.txt` |

---

## Notes

- All values shown in the dashboard, notebook, and report are computed
  directly from the dataset — no numbers are fabricated.
- The evaluation metrics (Precision@K, Recall@K, Hit Rate@K) reflect the
  actual sparsity of this dataset (many users, many products, relatively few
  interactions per product), so the raw scores are modest — this is
  realistic and expected for a first prototype, and is exactly the kind of
  result that motivates business actions like the ones listed in
  `report/project_report.md`.
