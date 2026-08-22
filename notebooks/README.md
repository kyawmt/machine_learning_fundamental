# Machine Learning Fundamentals — Companion Jupyter Notebook Suite

The interactive companion notebook suite for **Machine Learning Fundamentals — 4-Hour AI Engineer Interview Review**.

While the website explains the intuition, formulas, interview answers, and common mistakes, these notebooks do the one thing a static page cannot: **run the experiment and show the number.**

---

## ⚡ Quickstart

1. **Activate the virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r notebooks/requirements.txt
   ```

2. **(Optional) Run the website**:
   In a separate terminal from the repository root:
   ```bash
   npm run dev
   ```
   The interactive links `> 📖 **Website:** [...]` in every section link to `http://localhost:5173/`. The notebooks are completely self-contained and fully functional without running the website.

3. **Launch Jupyter Lab**:
   ```bash
   cd notebooks
   jupyter lab
   ```

---

## 📚 4-Hour Study Plan

| Notebook | Hour | Topics Covered | Runtime | Website Sections |
| :--- | :--- | :--- | :--- | :--- |
| [`00_start_here.ipynb`](00_start_here.ipynb) | Orientation | Environment check, dependency validation, synthetic dataset tour | ~10 min | [Study Plan](http://localhost:5173/#study-plan) |
| [`01_core_concepts.ipynb`](01_core_concepts.ipynb) | Hour 1 | Problem types, Splits, 6 Data Leakage experiments, Over/Underfitting, Empirical Bias–Variance, CV | ~60 min | `#problem-types`, `#splits`, `#data-leakage`, `#over-underfitting`, `#bias-variance`, `#cross-validation` |
| [`02_metrics_and_data.ipynb`](02_metrics_and_data.ipynb) | Hour 2 | Confusion matrix from scratch, Cost-sensitive thresholds, ROC vs PR-AUC sweep, Calibration, Scaling, Missingness, Categoricals, Class Imbalance | ~60 min | `#metrics`, `#feature-engineering`, `#feature-scaling`, `#missing-data`, `#categorical`, `#class-imbalance` |
| [`03_models_and_training.ipynb`](03_models_and_training.ipynb) | Hour 3 | GD from scratch (4 regimes), Logistic Regression from scratch, L1/L2 regularization paths, Hyperparameter tuning, 8-algorithm bake-off & RF tree correlation | ~60 min | `#regularization`, `#gradient-descent`, `#params-hyperparams`, `#algorithms` |
| [`04_review_and_practice.ipynb`](04_review_and_practice.ipynb) | Hour 4 | Accuracy vs Latency Pareto frontier, Bagging/Boosting variance/bias decomposition, PCA with/without scaling, K-Means from scratch, 11-step Capstone Pipeline, 8 Debugging Drills, Readiness Self-Assessment | ~60 min | `#model-selection`, `#ensembles`, `#dimensionality-reduction`, `#clustering`, `#pipeline`, `#interview-questions` |

---

## ⏱️ If You Only Have One Hour (Fast Path)

If interview preparation time is short, focus on the highest-yield empirical experiments:

1. **Notebook 01 — §1.3 Data Leakage 🔴**: Watch target leakage inflate AUC from 0.81 to 0.99, run drop-feature ablation, test temporal leakage, and see why SMOTE before split yields false confidence.
2. **Notebook 02 — §2.2 & §2.3 Cost Thresholds & PR-AUC 🔴**: Calculate the dollar savings of moving from 0.5 to the cost-optimal decision threshold, and watch PR-AUC collapse under extreme imbalance while ROC-AUC remains artificially flat.
3. **Notebook 04 — §4.6 Debugging Drills 🔴**: Debug 8 realistic, subtly flawed ML code blocks and see the quantified error cost of each bug.

---

## 🔬 What the Notebooks Prove vs Website Assertions

| Concept | The Website Asserts | The Notebook Proves & Measures |
| :--- | :--- | :--- |
| **Data Leakage** | "Split before any transformations" | Compares random split (0.93 AUC fiction) vs group split (0.84 AUC) vs temporal split (0.79 AUC honest future). |
| **Bias–Variance** | Conceptual U-shaped diagram | Empirically decomposes 200 bootstrap fits: \(\text{total} \approx \text{bias}^2 + \text{variance} + \sigma^2\) with exact numeric tables. |
| **Imbalance** | "PR-AUC is better than ROC-AUC under severe imbalance" | Sweeps positive rate from 50% down to 0.5%: ROC-AUC stays flat at 0.94 while PR-AUC collapses from 0.94 to 0.18. |
| **Random Forest** | "Decorrelating trees reduces ensemble variance" | Measures per-tree prediction correlation \(\rho\) across `max_features` and verifies \(\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2\). |
| **Feature Scaling** | "Tree models are scale invariant, distance models are not" | Executes `assert np.array_equal(tree_preds_scaled, tree_preds_unscaled)` while showing KNN/SVM accuracy degrade drastically without scaling. |
| **Regularization** | "L1 performs feature selection, L2 shrinks weights" | Plots exact coefficient paths vs \(\alpha\) on log-scale; demonstrates Lasso instability on collinear features vs Ridge stability. |
| **Production ML** | "Ship the pipeline, not the model weights" | Saves complete `ColumnTransformer` + `Pipeline` via `joblib`, reloads, and asserts bit-for-bit identical predictions. |

---

## 🛠️ Optional Dependencies & Fallbacks

All notebooks run on standard CPU without GPUs. Optional libraries (`imbalanced-learn`, `lightgbm`, `xgboost`, `optuna`, `ipywidgets`) are guarded by `try...except` blocks with automatic graceful fallbacks (e.g. falling back to `HistGradientBoostingClassifier` if LightGBM is not installed).
