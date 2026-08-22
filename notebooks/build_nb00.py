"""
Builder for 00_start_here.ipynb
"""

import nbformat as nbf


def create_nb00():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            """# 00 · Start Here — Setup, Environment & Dataset Tour 🔴

> 📖 **Website:** [4-Hour Study Plan](http://localhost:5173/#study-plan) · Orientation · 10 min

Welcome to the companion notebook suite for **Machine Learning Fundamentals — 4-Hour AI Engineer Interview Review**.

The website teaches intuition, formulas, comparison tables, interview answers, and common mistakes. **These notebooks do what a static page cannot: run the empirical experiments, decompose errors, break assumptions on purpose, and show the exact numbers.**

---

### 🌐 Pairing With the Website

The website is a fast Vite single-page application. While working through these notebooks, you can keep the website open on one screen and run the experiments on the other.
* To launch the website locally: run `npm run dev` from the repository root, then open `http://localhost:5173/`.
* Every major section in these notebooks contains a direct link back to its corresponding website topic.
* All notebooks are fully self-contained and run completely offline without internet or running the website.

---

### 🗺️ The 4-Hour Study Roadmap

| Notebook | Hour | Primary Focus | Key Experiments |
| :--- | :--- | :--- | :--- |
| **[`01_core_concepts.ipynb`](01_core_concepts.ipynb)** | Hour 1 | Core Foundations | Group/Temporal splits, 6 Data Leakage experiments, Empirical Bias-Variance decomposition, Nested CV |
| **[`02_metrics_and_data.ipynb`](02_metrics_and_data.ipynb)** | Hour 2 | Metrics & Data Prep | Confusion matrix from scratch, Cost-sensitive threshold tuning, ROC vs PR-AUC under severe imbalance, Calibration, Outliers |
| **[`03_models_and_training.ipynb`](03_models_and_training.ipynb)** | Hour 3 | Models & Training | Gradient Descent from scratch (4 regimes), Logistic regression from scratch, L1/L2 paths, 8-model bake-off & RF tree correlation |
| **[`04_review_and_practice.ipynb`](04_review_and_practice.ipynb)** | Hour 4 | Review & Practice | Accuracy/Latency Pareto frontier, PCA with/without scaling, K-Means from scratch, 11-step Capstone Pipeline, 8 Debugging Drills, Readiness Assessment |
"""
        ),
        nbf.v4.new_markdown_cell(
            """### 0.1 Environment Verification 🔴

**The question:** Are all required core libraries present with compatible versions so that no experiment fails midway through Hour 3?

**What you'll see:** A clear table checking Python, NumPy, Pandas, Matplotlib, and Scikit-Learn versions with confirmation indicators."""
        ),
        nbf.v4.new_code_cell(
            """import sys
import importlib

CORE_REQUIREMENTS = {
    "numpy": "1.26.0",
    "pandas": "2.1.0",
    "matplotlib": "3.8.0",
    "sklearn": "1.4.0",
}

print(f"Python Version: {sys.version.split()[0]}")
assert sys.version_info >= (3, 9), "Python 3.9+ is required."

all_passed = True
for pkg, min_ver in CORE_REQUIREMENTS.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"✅ {pkg:<12} {ver:<10} (minimum: {min_ver})")
    except ImportError:
        print(f"❌ {pkg:<12} NOT INSTALLED (required: pip install {pkg}>={min_ver})")
        all_passed = False

if all_passed:
    print("\\nAll core dependencies are satisfied and ready for execution.")
"""
        ),
        nbf.v4.new_markdown_cell(
            """> 🎤 **In an interview:** "I always establish an explicit environment baseline and deterministic seeds before running benchmarks, ensuring experiment reproducibility across team members." """
        ),
        nbf.v4.new_markdown_cell(
            """### 0.2 Optional Dependencies Check 🟠

**The question:** Are optional acceleration libraries (`lightgbm`, `xgboost`, `imbalanced-learn`, `optuna`, `ipywidgets`) available?

**What you'll see:** Status of optional packages, with confirmation that graceful fallbacks are enabled if any are absent."""
        ),
        nbf.v4.new_code_cell(
            """OPTIONAL_PACKAGES = ["imblearn", "lightgbm", "xgboost", "optuna", "ipywidgets"]

print("Checking optional packages (notebooks degrade gracefully if missing):\\n")
for pkg in OPTIONAL_PACKAGES:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"✅ Optional: {pkg:<16} {ver}")
    except ImportError:
        print(f"ℹ️ Optional: {pkg:<16} not found (using standard scikit-learn fallbacks)")
"""
        ),
        nbf.v4.new_markdown_cell(
            """> 🎤 **In an interview:** "For production pipelines, we use clean interfaces with fallback capabilities—for instance, falling back from LightGBM to HistGradientBoosting if native C++ bindings are restricted in an execution container." """
        ),
        nbf.v4.new_markdown_cell(
            """### 0.3 Synthetic Churn Panel Tour 🔴
> 📖 **Website:** [Data Leakage](http://localhost:5173/#data-leakage)

**The question:** What does our recurring synthetic dataset look like, and what specific ML traps are planted inside it?

**What you'll see:** A preview of `make_churn_panel()`, including its shape, columns, missingness pattern, class imbalance, and a guided breakdown of each planted property."""
        ),
        nbf.v4.new_code_cell(
            """from mlprep.data import make_churn_panel

# Generate the recurring churn panel (4,000 users across 3 monthly snapshots = 12,000 rows)
df_churn = make_churn_panel(n_users=4000, snapshots_per_user=3, seed=7)

print(f"Dataset Shape: {df_churn.shape[0]:,} rows × {df_churn.shape[1]} columns")
print(f"Unique Users  : {df_churn['user_id'].nunique():,}")
print(f"Snapshots     : {df_churn['snapshot_date'].dt.strftime('%Y-%m-%d').unique().tolist()}")
print(f"Churn Rate    : {df_churn['churned'].mean():.2%} (severe class imbalance)")
print(f"Missing Income: {df_churn['income'].isna().mean():.2%} (MNAR missingness)")

display_cols = ["user_id", "snapshot_date", "tenure_days", "monthly_spend", "logins_30d", "income", "plan_region", "device", "support_tickets_30d", "cancellation_reason_code", "churned"]
df_churn[display_cols].head(6)
"""
        ),
        nbf.v4.new_markdown_cell(
            """#### 🔎 Planted Properties in the Churn Panel:

| Column | Planted Mechanism & Educational Purpose |
| :--- | :--- |
| `user_id` | Repeated records per user across monthly snapshots $\\rightarrow$ Demonstrates **group leakage** and the necessity of `GroupKFold`. |
| `snapshot_date` | Monotone chronological timestamp $\\rightarrow$ Demonstrates **temporal leakage** and forward-chaining validation (`TimeSeriesSplit`). |
| `tenure_days`, `monthly_spend`, `logins_30d` | Wildly disparate scales ($10\\text{--}2000$, $\\$15\\text{--}350$, $0\\text{--}100$) $\\rightarrow$ Demonstrates **feature scaling** necessity for distance/linear models and invariance in trees. |
| `income` | Missing Not At Random (**MNAR**) missingness (missing at $>50\\%$ rate for high earners) $\\rightarrow$ Demonstrates **missing indicator** features. |
| `plan_region` | High cardinality ($\sim 200$ levels) $\\rightarrow$ Demonstrates **one-hot explosion vs out-of-fold target encoding**. |
| `device` | 4 unordered categories (`iOS`, `Android`, `Web`, `Desktop`) $\\rightarrow$ Demonstrates the **ordinal encoding false-order trap**. |
| `support_tickets_30d` | Features collected over a window straddling the churn label $\\rightarrow$ Demonstrates **temporal leakage**. |
| `cancellation_reason_code` | Only populated for churned customers $\\rightarrow$ Demonstrates **target leakage** and drop-feature ablation diagnostics. |
| `churned` | $\\sim 6\\%$ positive baseline prevalence $\\rightarrow$ Demonstrates **ROC-AUC vs PR-AUC divergence**, cost-sensitive thresholds, and resampling methods. |
"""
        ),
        nbf.v4.new_markdown_cell(
            """> 🎤 **In an interview:** "Before training any model, I run an exploratory audit on entity identifiers, timestamp monotonicity, and target-conditional missingness to catch leakage before it reaches cross-validation." """
        ),
        nbf.v4.new_markdown_cell(
            """### 0.4 How to Use These Notebooks

1. **Run All Top-to-Bottom**: At the beginning of each hour, run the notebook top-to-bottom to confirm the baseline.
2. **Attempt the Exercises**: Try solving the exercises in the `# YOUR CODE HERE` blocks before expanding the collapsible `💡 Solution` dropdowns.
3. **Say the 🎤 Interview Lines Out Loud**: The interview callout at the end of each cell summarizes the exact phrasing Senior and Staff ML engineers use in technical interviews.

Ready to begin? Move to **[`01_core_concepts.ipynb`](01_core_concepts.ipynb)**!"""
        ),
    ]

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = create_nb00()
    with open("notebooks/00_start_here.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 00_start_here.ipynb")
