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
| [`02_metrics_and_data.ipynb`](02_metrics_and_data.ipynb) | Hour 2 | Confusion matrix, thresholds/capacity, ROC vs PR-AUC, calibration, feature interactions, scaling, missingness, categoricals, imbalance | ~60 min | `#metrics`, `#feature-engineering`, `#feature-scaling`, `#missing-data`, `#categorical`, `#class-imbalance` |
| [`03_models_and_training.ipynb`](03_models_and_training.ipynb) | Hour 3 | GD from scratch (4 regimes), Logistic Regression from scratch, L1/L2 regularization paths, Hyperparameter tuning, 8-algorithm bake-off & RF tree correlation | ~60 min | `#regularization`, `#gradient-descent`, `#params-hyperparams`, `#algorithms` |
| [`04_review_and_practice.ipynb`](04_review_and_practice.ipynb) | Hour 4 | Measured Pareto frontier, ensemble bias/variance, PCA failure case, K-Means/DBSCAN, 11-step capstone, monitoring, 8 debugging drills | ~60 min | `#model-selection`, `#ensembles`, `#dimensionality-reduction`, `#clustering`, `#pipeline`, `#interview-questions` |

---

## ⏱️ If You Only Have One Hour (Fast Path)

If interview preparation time is short, focus on the highest-yield empirical experiments:

1. **Notebook 01 — §1.3 Data Leakage 🔴**: Measure target, temporal, duplicate-row, target-encoding, preprocessing, and pre-CV SMOTE leakage against like-for-like clean references.
2. **Notebook 02 — §2.2 & §2.3 Cost Thresholds & PR-AUC 🔴**: Calculate the dollar saving from a validation-chosen threshold and watch PR-AUC respond to prevalence while ROC-AUC remains stable in expectation.
3. **Notebook 04 — §4.5 & §4.7 Capstone and Debugging 🔴**: Walk through a sealed train/validation/test lifecycle, then debug eight realistic ML evaluation defects.

---

## 🔬 What the Notebooks Prove vs Website Assertions

| Concept | The Website Asserts | The Notebook Proves & Measures |
| :--- | :--- | :--- |
| **Data Leakage** | "Split before any transformations" | Counts shared users under row splits and measures six leakage mechanisms against clean references. |
| **Bias–Variance** | Conceptual U-shaped diagram | Empirically decomposes 200 bootstrap fits: \(\text{total} \approx \text{bias}^2 + \text{variance} + \sigma^2\) with exact numeric tables. |
| **Imbalance** | "Use more than accuracy" | Separates ranking, calibration, threshold decisions, dollar cost, recall, and precision at a fixed review capacity. |
| **Random Forest** | "Decorrelating trees reduces ensemble variance" | Measures per-tree prediction correlation across genuinely distinct `max_features` settings and verifies the variance-of-an-average identity. |
| **Feature Scaling** | "Scaling depends on the model" | Fits scalers on training data only, checks exact seeded-forest prediction equality here, and compares Standard/MinMax/Robust scaling under an outlier. |
| **Regularization** | "L1 performs feature selection, L2 shrinks weights" | Plots exact coefficient paths vs \(\alpha\) on log-scale; demonstrates Lasso instability on collinear features vs Ridge stability. |
| **Production ML** | "Ship the pipeline, not the model weights" | Saves complete `ColumnTransformer` + `Pipeline` via `joblib`, reloads, and asserts bit-for-bit identical predictions. |

To regenerate every notebook and fail on the first broken cell, run this from the repository root:

```bash
python notebooks/run_all_notebooks.py
```

---

## 🛠️ Optional Dependencies & Fallbacks

All notebooks run on standard CPU without GPUs. `imbalanced-learn` adds the optional resampling rows; when it is absent those rows are skipped. The other packages reported by Notebook 00 are optional extensions and are not required for the core path.
