Build a **companion Jupyter notebook suite** for the existing website in this repository — *Machine Learning Fundamentals — 4-Hour AI Engineer Interview Review*.

The website already teaches the concepts: intuition, formulas, comparison tables, interview answers, common mistakes, and recall drills. **The notebooks must not repeat that.** Their job is the one thing a static page cannot do: **run the experiment and show the number.**

The target learner already knows Python and pandas, has studied ML before, and is preparing for an **AI Engineer / ML Engineer** interview. They will have the website open on one screen and the notebook on the other.

## Primary Goal

Create a notebook suite that turns the website's claims into **things the learner has personally observed run**, in roughly the same four hours.

Every notebook section must do at least one of these:

* **Measure** something the website only asserts (empirically decompose bias and variance; measure tree correlation inside a random forest).
* **Break** something on purpose and show the inflated number, then fix it and show the honest number (leakage, wrong splitter, SMOTE before the split).
* **Implement** something from scratch so the formula stops being a formula (gradient descent, log loss, precision/recall, k-means).
* **Compare** things under identical conditions so the tradeoff is a table, not an opinion.

If a cell does none of those, delete it.

---

# 1. Deliverables

Create a `notebooks/` directory:

```
notebooks/
  00_start_here.ipynb        index, setup check, dataset tour  (~10 min)
  01_core_concepts.ipynb     Hour 1                            (~60 min)
  02_metrics_and_data.ipynb  Hour 2                            (~60 min)
  03_models_and_training.ipynb Hour 3                          (~60 min)
  04_review_and_practice.ipynb Hour 4                          (~60 min)
  mlprep/
    __init__.py
    data.py                  synthetic dataset generators
    plots.py                 shared matplotlib style + plot helpers
    checks.py                assertion helpers used by exercises
  requirements.txt
  README.md
```

**Five notebooks, not one.** A single four-hour notebook is unusable: kernel restarts lose everything, scrolling is hopeless, and you cannot run one hour without paying for the others. One notebook per study hour matches how the website is structured and how the learner will actually work.

Also update the repository root `README.md` with a short **"Study with the notebooks"** section pointing at `notebooks/README.md`.

---

# 2. Pairing With the Website

Every notebook section must map to a website section, and say so.

Open every notebook section's first markdown cell with a link line in exactly this format:

```markdown
> 📖 **Website:** [Data Leakage](http://localhost:5173/#data-leakage) · Hour 1 · 12 min
```

The site is a Vite single-page app, so these links only resolve while the dev server is
running (`npm run dev` from the repo root). Say so once, in Notebook 00 and in
`notebooks/README.md` — do not repeat the caveat on every section. Define the base URL
as a single constant in `mlprep/__init__.py` (`SITE = "http://localhost:5173/"`) so it
can be changed in one place if the site is later deployed.

Use these anchors (they are the real section ids in `src/data/sections.ts`):

| Notebook | Website section | Anchor |
| --- | --- | --- |
| 01 | ML Problem Types | `#problem-types` |
| 01 | Train / Validation / Test Split | `#splits` |
| 01 | Data Leakage | `#data-leakage` |
| 01 | Overfitting and Underfitting | `#over-underfitting` |
| 01 | Bias–Variance Tradeoff | `#bias-variance` |
| 01 | Cross-Validation | `#cross-validation` |
| 02 | Model Evaluation Metrics | `#metrics` |
| 02 | Feature Engineering | `#feature-engineering` |
| 02 | Feature Scaling | `#feature-scaling` |
| 02 | Handling Missing Data | `#missing-data` |
| 02 | Categorical Variables | `#categorical` |
| 02 | Class Imbalance | `#class-imbalance` |
| 03 | Regularization | `#regularization` |
| 03 | Gradient Descent | `#gradient-descent` |
| 03 | Hyperparameters vs Parameters | `#params-hyperparams` |
| 03 | Common ML Algorithms | `#algorithms` |
| 04 | Model Selection | `#model-selection` |
| 04 | Ensemble Methods | `#ensembles` |
| 04 | Dimensionality Reduction | `#dimensionality-reduction` |
| 04 | Clustering Fundamentals | `#clustering` |
| 04 | ML Pipeline | `#pipeline` |
| 04 | Common Interview Questions | `#interview-questions` |

Keep the same **priority labels** the website uses, in each section heading:

* 🔴 **Must Know** — roughly 70%
* 🟠 **Important** — roughly 25%
* 🟢 **Good to Know** — roughly 5%

Where the website has an interactive lab (Threshold Lab, Overfitting Lab, PCA Lab, Bias–Variance Lab, K-Means Lab, Regularization Lab, Gradient Descent Lab), the notebook should **reconstruct it in code and then go one step further than the slider allows** — sweep the parameter programmatically, or measure something the lab only shows qualitatively.

---

# 3. One Recurring Dataset

Do **not** invent a new toy dataset every cell. Build one synthetic dataset in `mlprep/data.py` and reuse it across all four notebooks so the learner builds familiarity instead of re-orienting constantly.

```python
make_churn_panel(n_users=4000, snapshots_per_user=3, seed=7) -> pd.DataFrame
```

It must be deliberately constructed to support the demos, and every property below must be **documented in the docstring** so the learner knows what was planted:

| Column | Purpose it serves |
| --- | --- |
| `user_id` | repeated rows per user → **group leakage** demo, `GroupKFold` |
| `snapshot_date` | monotone time index → **temporal split** demo, `TimeSeriesSplit` |
| `tenure_days`, `monthly_spend`, `logins_30d` | wildly different scales → **scaling** demo |
| `income` | **MNAR** missingness (missing more often at the high end) |
| `plan_region` | ~200 levels → **high-cardinality / target encoding** demo |
| `device` | 4 unordered levels → **one-hot vs ordinal** demo |
| `support_tickets_30d` | window straddles the label → **temporal leakage** demo |
| `cancellation_reason_code` | only populated for churners → **target leakage** demo |
| `churned` | ~6% positive rate → **imbalance** demo |

Requirements:

* Fully deterministic given the seed. The learner and the author must see identical numbers.
* A **real, learnable signal** underneath — a model with the leaky columns dropped should reach roughly 0.78–0.84 ROC-AUC. If it reaches 0.99, the dataset is broken; if it reaches 0.55, none of the demos will show anything.
* Include at least one genuine **interaction** (e.g. spend matters much more for short-tenure users) so tree models beat linear models for an honest reason.
* Fast: generating it must take under a second.

Add two smaller helpers:

* `make_poly_sample(n=14, noise=0.16, seed=344)` — the 1-D regression sample used for the overfitting/bias–variance work. **Use exactly these defaults so the notebook reproduces the website's Overfitting Lab curve.**
* `make_scored_population(n=1000, positive_rate=0.03, seed=20260822)` — pre-scored positives/negatives for metric work. **Match the website's Threshold Lab** (ROC-AUC ≈ 0.94, PR-AUC ≈ 0.41), so the learner sees the same numbers they saw in the browser.

Otherwise use only **scikit-learn's bundled datasets** — `load_breast_cancer`, `load_diabetes`, `load_wine`, `load_digits`, `load_iris`. **No network downloads.** No `fetch_*`. No Kaggle CSVs. The notebooks must run on a plane.

---

# 4. Cell Structure

Every experiment follows the same five-part rhythm. Be strict about this — the consistency is what makes the suite skimmable.

### 1. Markdown: heading + website link + the question

```markdown
### 1.3.2 Target leakage 🔴
> 📖 **Website:** [Data Leakage](http://localhost:5173/#data-leakage)

**The question:** you add a feature and AUC jumps from 0.79 to 0.97. Is that a
breakthrough or a bug? Let's cause it on purpose so you recognise it instantly.
```

### 2. Markdown: "What you'll see"

One or two sentences predicting the outcome **before** it runs. This turns passive execution into a prediction the learner can be wrong about, which is the entire point.

### 3. Code

Short. Under ~25 lines. Push repetition into `mlprep/`. No cell should take more than ~10 seconds.

### 4. Output that interprets itself

Never print a bare array or a raw `.head()` and move on. End every cell with a formatted line that states the finding:

```python
print(f"clean features   : ROC-AUC {clean:.3f}")
print(f"+ leaky feature  : ROC-AUC {leaky:.3f}   <-- {leaky-clean:+.3f}")
print(f"\nA jump of {leaky-clean:+.3f} from one column is not a modelling win.")
```

### 5. Markdown: 🎤 In an interview

The one sentence to say out loud, written the way a person speaks. This is the bridge back to the website.

```markdown
> 🎤 **In an interview:** "A jump that large from a single feature is almost
> never real. I'd trace the column to its source and establish when its value
> is written relative to the prediction timestamp."
```

---

# 5. Notebook 00 — Start Here

Short, and it must work before anything else does.

* Title, what the suite is, how it pairs with the website, the four-hour plan as a table with links to each notebook.
* **Environment check cell** — print Python and library versions, assert minimum versions, and print a clear ✅/❌ per package with the exact `pip install` line to fix a failure. Never let the learner discover a missing package in the middle of Hour 3.
* **Optional-dependency check** — `lightgbm` / `xgboost`. If absent, print a friendly note that Notebook 03 will fall back to `HistGradientBoostingClassifier` and continue. Nothing may hard-fail on a missing optional package.
* **Dataset tour** — generate `make_churn_panel()`, show `.head()`, dtypes, missingness bar, the class balance, and a one-line description of every planted property from §3. The learner should finish this cell knowing exactly what traps are in the data.
* **How to use these notebooks** — Run All at the top of each hour, then work down; do the exercises before revealing solutions; say the 🎤 lines out loud.

---

# 6. Notebook 01 — Core Concepts (Hour 1)

### 1.1 Problem types 🔴 (5 min)

* Take one raw quantity and frame it **both ways** — predict `monthly_spend` (regression) vs `is_high_spender` (classification). Fit both, show that the metrics are not comparable and that thresholding a regressor is not the same model as a classifier.
* One short cell: given four target columns, print the problem type and the metric you'd reach for.

### 1.2 Splits 🔴 (10 min)

**The highest-value section in this notebook.** Same data, same model, four splitters, four very different scores:

```
random split          ROC-AUC 0.93   <- fiction
stratified split      ROC-AUC 0.93   <- still fiction, groups leak
group split (user)    ROC-AUC 0.84   <- honest about new users
time split (future)   ROC-AUC 0.79   <- honest about the future
```

* Print a table exactly like that, with a sentence under it naming which number you would report to a stakeholder and why.
* Show the mechanism, don't just assert it: count how many `user_id`s appear in **both** train and test under a random split.
* Show the class-ratio drift across folds with `KFold` vs `StratifiedKFold` on the 6% target.
* **Exercise:** write `split_honestly(df)` that is simultaneously time-respecting and group-respecting. Assert no `user_id` crosses the boundary and no train `snapshot_date` is after any test date.

### 1.3 Data leakage 🔴 (15 min)

The centrepiece. Five sub-experiments, each with the same shape: *the seductive number → the mechanism → the honest number.* Collect all five into one summary table at the end.

1. **Scale before split** vs `Pipeline`. Be honest that the gap here is small — and explain that the rule exists because the *same mistake* with the transforms below is severe.
2. **Target leakage** — add `cancellation_reason_code`, watch AUC jump, then run a **drop-the-top-feature ablation** and show performance collapse. Teach the ablation as the diagnostic.
3. **Temporal leakage** — `support_tickets_30d` computed over a window that straddles the label vs computed strictly as-of the prediction timestamp.
4. **Duplicate rows** across the split.
5. **Naive target encoding** vs out-of-fold target encoding on `plan_region`.
6. **SMOTE before the split** vs inside an `imblearn` pipeline. Show CV score inflating while the test score does not move — and make the point that leakage under CV produces a *tight, confident, wrong* estimate.

End with a **leakage audit function** the learner writes:

```python
def audit(df, target, time_col, group_col) -> pd.DataFrame:
    """One row per feature: correlation with target, % missing,
    whether it is constant within a class, and a leakage suspicion flag."""
```

### 1.4 Overfitting and underfitting 🔴 (10 min)

* Reproduce the website's Overfitting Lab: `make_poly_sample()`, sweep degree 1–12, plot train and validation RMSE. **The learner should see the same U-shaped curve they dragged the slider across, with the minimum at degree 5.**
* Then go beyond it: `learning_curve` on the churn data. Read off whether more data will help — if the validation curve is still rising, it will; if flat, it will not. Say that explicitly.
* Tree depth sweep with the same diagnostic.
* **Exercise:** given three (train score, val score) pairs, print the diagnosis and the first two things you'd try.

### 1.5 Bias–variance 🔴 (12 min)

The section that most justifies the notebook existing. The website shows an *illustrative* curve; here you **measure the real decomposition**.

* Draw 200 bootstrap training sets. Fit the model on each. For a grid of test points, compute:
  * `bias² = (mean prediction − true function)²`
  * `variance = var(predictions across the 200 fits)`
  * verify `total ≈ bias² + variance + noise`
* Do it for three model complexities (degree 1, 5, 12 — or tree depth 2, 6, None) and plot all three decompositions side by side.
* Print the numeric decomposition table. Seeing `bias²=0.41 var=0.02` become `bias²=0.01 var=0.58` is worth more than any diagram.
* Show the 200 fitted curves overlaid at low and high complexity — the visual spread *is* the variance.

### 1.6 Cross-validation 🔴 (8 min)

* Same estimator, five splitters, report `mean ± std`. Emphasise the std, not the mean.
* Demonstrate why the wrong splitter lies, using the group and time structure from 1.2.
* **Nested CV** vs a flat tuned CV score on a small subsample — show the optimism gap, and connect it to the website's answer to "if you're using cross-validation, do you still need a test set?"
* Show the multiple-comparisons effect concretely: evaluate 500 random configurations on a small validation set and plot `best-so-far CV score` against `its test score`. The two curves diverging is the whole lesson.

---

# 7. Notebook 02 — Metrics and Data Prep (Hour 2)

### 2.1 Confusion matrix from scratch 🔴 (8 min)

* Implement `precision`, `recall`, `specificity`, `f1` with numpy only. `assert np.isclose(...)` against `sklearn.metrics`. A green assertion is the reward.
* Pretty-print a labelled confusion matrix with TP/FP/FN/TN in the right cells and both marginals.
* **Exercise:** from a printed confusion matrix, compute all four by hand in a markdown cell, then check with the functions.

### 2.2 Thresholds and cost 🔴 (12 min)

* Reproduce the website's Threshold Lab with `make_scored_population()` — same 1,000 cases, same 3% base rate, same ROC-AUC ≈ 0.94 / PR-AUC ≈ 0.41.
* Plot precision, recall and F1 against threshold on one axis. Point out where precision is **non-monotonic** in the sparse tail — the website mentions it; here the learner sees the wobble.
* **The interview-winning cell:** define a cost matrix (`cost_fp = $8`, `cost_fn = $400`), compute expected cost across all thresholds, and pick the argmin. Print the chosen threshold, the resulting confusion matrix and the dollar saving vs the default 0.5. Then change the costs and rerun to show the choice move.
* `precision_at_k` for a fixed review capacity (100/day), and why that beats a global metric when capacity is the real constraint.

### 2.3 ROC-AUC vs PR-AUC under imbalance 🔴 (10 min)

Hold the model fixed, vary the base rate from 50% down to 0.5% by subsampling negatives, and plot both AUCs against the base rate.

ROC-AUC will be nearly flat; PR-AUC will fall off a cliff. **That single plot is the most convincing argument for PR-AUC that exists**, and it is impossible to put on a static page.

Also plot the PR curve's no-skill baseline moving with the base rate.

### 2.4 Calibration 🟠 (8 min)

* Reliability diagram + Brier score for logistic regression, random forest and naive Bayes on the same data. Rank them by AUC and by Brier and show the rankings disagree.
* Show `class_weight="balanced"` **wrecking** calibration while leaving AUC untouched, then repair it with `CalibratedClassifierCV` on held-out data.
* Tie to the website's line: ranking-only use cases don't care; expected-value calculations do.

### 2.5 Regression metrics 🔴 (8 min)

* MAE vs RMSE on clean data, then inject three outliers and recompute. Show the RMSE/MAE ratio as an outlier detector.
* Empirically verify the optimal constant predictor: minimise MAE → the median; minimise MSE → the mean. Do it by brute-force search over candidate constants and plot both loss curves.
* Produce a **negative test R²** on purpose with a badly overfitted model, and explain what it means.

### 2.6 Scaling 🔴 (7 min)

* Same pipeline, `scaler` vs `passthrough`, across KNN / SVM / logistic regression / random forest / gradient boosting. One results table.
* The punchline cell: `assert np.array_equal(tree_preds_scaled, tree_preds_unscaled)` — **exactly** equal, not approximately. Then one sentence on why: splits depend only on value ordering.
* Show StandardScaler vs MinMaxScaler vs RobustScaler on a feature with outliers, as three histograms.

### 2.7 Missing data 🟠 (8 min)

* Simulate MCAR and MNAR missingness on the same column and impute both with the median. Show the MCAR case survives and the MNAR case loses signal.
* Add `add_indicator=True` and show the MNAR performance recover. Print the model's coefficient on the indicator — the size of that coefficient is the missingness *being* the signal.
* Show a GBM handling NaN natively with no imputer at all.

### 2.8 Categorical variables 🔴 (8 min)

* Ordinal-encode `device` (unordered) and fit both a linear model and a tree. The linear model degrades; the tree barely notices. Explain why in one line.
* One-hot vs frequency vs out-of-fold target encoding on `plan_region` (200 levels): score, feature count, fit time in one table.
* Reproduce the naive-target-encoding leak from 1.3 to reinforce it.
* Handle an **unseen category at inference** — show the crash, then `handle_unknown="ignore"`.

### 2.9 Class imbalance 🔴 (10 min)

Run the website's recommended order and build the results table that proves the ordering is right:

| Strategy | PR-AUC | Precision@100 | Recall | Notes |
| --- | --- | --- | --- | --- |
| baseline @ 0.5 | | | | |
| + tuned threshold | | | | free, no retrain |
| + class_weight | | | | |
| + SMOTE (in-fold) | | | | |
| under-sampling | | | | |

Include a column for wall-clock fit time. The expected finding — threshold tuning and class weights do most of the work for almost none of the cost — is one of the most useful things a candidate can say, and here they will have measured it.

Finish by showing the **probability distortion** resampling causes, and recalibrating.

---

# 8. Notebook 03 — Models and Training (Hour 3)

### 3.1 Gradient descent from scratch 🔴 (12 min)

* Implement linear-regression GD in ~10 lines of numpy. Print the loss every N steps.
* Reproduce the website's Gradient Descent Lab: run the same learning rates and confirm the four regimes — slow, healthy, oscillating-but-converging, divergent. **Show the actual `nan` appearing.**
* Plot the parameter trajectory over a 2-D loss contour for batch vs mini-batch vs SGD. The noise in the SGD path is the entire point.
* Verify convergence against the closed-form OLS solution with an assertion.
* **Exercise:** add momentum in three lines and show it escaping a ravine faster.

### 3.2 Logistic regression from scratch 🔴 (8 min)

* Sigmoid, log loss, and the gradient `X.T @ (p - y) / n`. Train it. Assert the coefficients match `LogisticRegression(penalty=None)` to 2 decimals.
* Demonstrate **why not MSE**: plot both losses as a function of a single weight and show MSE's flat, saturated region where the gradient dies.
* Exponentiate a coefficient and interpret it as an odds ratio in a printed sentence.

### 3.3 Regularization 🔴 (12 min)

* Coefficient paths: plot every coefficient against `alpha` on a log axis for Ridge and Lasso, side by side. Lasso's paths hitting exactly zero, one at a time, is the money shot.
* Count nonzero coefficients vs alpha.
* **Correlated-feature instability:** duplicate a feature with slight noise, refit Lasso across several bootstrap resamples, and show it arbitrarily picking one or the other. Then show Ridge splitting the weight between them. This is the concrete version of a claim the website can only state.
* Show the penalty is **unit-dependent**: multiply one feature by 1000, refit without scaling, and watch its coefficient get crushed.
* Validation curve over alpha; mark the minimum and the **one-standard-error** choice.

### 3.4 Parameters vs hyperparameters 🟠 (5 min)

* Fit a model, then programmatically list `get_params()` (what you chose) against the learned attributes ending in `_` (what the data chose). Two printed columns; the distinction becomes obvious and unforgettable.

### 3.5 Grid vs random search 🔴 (8 min)

* Construct a problem where **one hyperparameter dominates and another is irrelevant**.
* Run a 5×5 grid and a 25-draw random search — same budget — and print how many *distinct values of the important hyperparameter* each explored (5 vs 25).
* Plot both searches as scatter points in the 2-D hyperparameter space, coloured by score. The picture makes the argument better than any paragraph.
* Add `optuna` **only if already installed**; otherwise print a short note describing what Bayesian optimization would add and move on.

### 3.6 Algorithm bake-off 🔴 (15 min)

One function, eight algorithms, **identical CV splits**:

```python
bake_off(models, X, y, cv) -> pd.DataFrame
# columns: model, cv_mean, cv_std, fit_seconds, predict_ms_per_1k,
#          needs_scaling, handles_nan, n_params
```

Cover linear/logistic regression, decision tree, random forest, gradient boosting, KNN, SVM, naive Bayes. Sort by score, then **discuss the rows where the best score is not the best choice**.

Then per-algorithm micro-experiments, each one short:

* **Decision tree** — print the tree, compute the root Gini by hand, assert it matches `tree_.impurity[0]`.
* **Random forest** — OOB score vs CV score; then **measure the correlation between per-tree predictions** at `max_features=1`, `sqrt`, and `all`, and connect it to the `ρσ² + (1−ρ)σ²/B` formula on the website. Show variance falling as ρ falls. This is the best experiment in the notebook.
* **Gradient boosting** — plot train and validation loss per boosting round using `staged_predict`; find the crossover; confirm early stopping lands there.
* **KNN** — measure the nearest/farthest distance ratio as dimensionality grows from 2 to 200. Watch it converge to 1. That *is* the curse of dimensionality, measured.
* **SVM** — a 3×3 grid of C × gamma with decision-boundary plots on 2-D data, plus support-vector counts. Under- and overfitting become visible.
* **Naive Bayes** — fit on text features, show excellent accuracy alongside probabilities pinned at 0.999. Good argmax, terrible calibration.

---

# 9. Notebook 04 — Review and Practice (Hour 4)

### 4.1 Model selection under constraints 🟠 (6 min)

* Extend the bake-off table with inference latency and model size on disk.
* Plot the **accuracy vs latency Pareto frontier** and mark which models are dominated.
* Print the sentence a candidate should say: "within a 10ms budget the frontier is X; the extra 0.4% AUC from Y costs 40ms."

### 4.2 Ensembles 🔴 (10 min)

* **Measure bagging's variance reduction**: fit one deep tree on 100 bootstrap samples, measure prediction variance, then measure it for the averaged ensemble. Print both numbers and the ratio.
* **Measure boosting's bias reduction**: track ensemble bias per round.
* **Stacking, done wrong then right**: in-sample base predictions vs out-of-fold. Show the wrong version scoring beautifully in CV and collapsing on test.
* Soft vs hard voting on the same base models.

### 4.3 PCA 🟠 (8 min)

* Cumulative explained-variance curve; pick the 95% cut programmatically.
* **PCA with and without standardization** on features with mismatched units. Print PC1's loadings both ways — this reproduces the website's PCA Lab as numbers rather than arrows.
* The failure case the website warns about: construct data where the **lowest-variance direction is the one that predicts y**, and show PCA discarding it. Then show a supervised alternative doing better.
* Timing: model fit time before vs after PCA.

### 4.4 Clustering 🟢 (8 min)

* Implement k-means from scratch — assign, recompute, repeat — and assert convergence to the same inertia as sklearn.
* Elbow and silhouette side by side; note that silhouette has a real maximum and inertia doesn't.
* **Failure gallery:** two moons, unequal variances, unequal cluster sizes. K-means fails on all three; DBSCAN succeeds on the first. Show it rather than saying it.
* Initialization sensitivity: run `n_init=1` ten times and print the spread of final inertias.

### 4.5 Capstone: end-to-end pipeline 🔴 (15 min)

The "walk me through building a model from raw data to production" question, written as code. One clean, commented, top-to-bottom cell block:

1. Frame the problem — target definition, prediction timestamp, cost of each error type, chosen metric. In markdown, before any code.
2. Load, clean, and run the leakage audit from §1.3.
3. Split honestly (time + group), and seal the test set.
4. `ColumnTransformer` — numeric impute+scale, low-cardinality one-hot, high-cardinality target-encode — all inside a `Pipeline`.
5. Baseline (majority class, then logistic regression).
6. Two candidates, cross-validated on identical splits.
7. Tune with a modest randomized search.
8. Choose the threshold from the cost matrix.
9. **Open the test set exactly once.** Print the final metrics with a bootstrap confidence interval.
10. Persist the whole pipeline with `joblib`, reload it, and assert the reloaded pipeline reproduces the predictions bit for bit — the concrete version of "ship the pipeline, not the model."
11. Emit a small **model card**: data window, features, metric, threshold, known limitations, retraining trigger.

Then a **train/serve skew** demo: score a single incoming row through the saved pipeline, then score it again with a subtly different manual preprocessing path, and show the predictions diverge.

### 4.6 Debugging drills 🔴 (10 min)

Eight cells, each containing plausible, working code with exactly one defect. The learner must find it before revealing the answer.

1. Scaler fitted before the split.
2. `KFold` on time-series data.
3. SMOTE applied to the test set.
4. Hyperparameters tuned on the test set.
5. Two models compared on different random splits.
6. Accuracy reported on a 1% positive rate.
7. Target encoding computed on the full dataset.
8. `fit_transform` called on the validation set.

Each drill gets: the code, a "what's wrong?" prompt, then a collapsed solution with **the fix and the size of the error it was causing**. Numbers, not just prose.

### 4.7 Self-assessment 🔴 (5 min)

Twelve statements, code-flavoured, mirroring the website's readiness check but about *doing* rather than *explaining*:

* "I can write a split that respects both time and group structure from memory."
* "I can pick a decision threshold from a cost matrix without looking it up."
* "I can implement precision, recall and F1 from a confusion matrix."
* "I can write a gradient-descent loop from scratch."
* "I can build a `ColumnTransformer` that handles numeric, low- and high-cardinality features."
* "I can spot leakage in someone else's notebook."
* …and six more.

Implement it as an interactive checklist (`ipywidgets` if available, plain markdown checkboxes otherwise) that prints a score and, for each unticked item, the notebook section to revisit.

Close with the website's closing line so the two artefacts end together:

> **If you can explain every red-priority topic without looking at the notes, you are ready for the Machine Learning fundamentals portion of an AI Engineer interview.**

---

# 10. Exercises

At least **twelve** exercises across the suite, weighted toward Notebooks 01 and 02.

Format, used consistently:

```python
# ── Exercise 1.2 ────────────────────────────────────────────────
# Write split_honestly(df) so that no user_id appears in both train
# and test, AND every training snapshot_date precedes every test one.
#
# Return (train_df, test_df).

def split_honestly(df, test_frac=0.2):
    # YOUR CODE HERE
    raise NotImplementedError

check_1_2(split_honestly)   # from mlprep.checks — prints ✅ or a specific hint
```

Then, in the next markdown cell:

````markdown
<details>
<summary>💡 Solution</summary>

```python
...
```

**Why this works:** ...
</details>
````

Requirements:

* `mlprep/checks.py` must give a **specific, actionable** failure message — "3 user_ids appear in both splits" beats "assertion failed".
* Solutions must be genuinely idiomatic, not deliberately clumsy.
* Every exercise must be solvable from what the notebook has already shown.

---

# 11. Plotting Standards

Put a single `mlprep.plots.use_style()` call in each notebook's setup cell.

* **Matplotlib only.** No seaborn, plotly, or bokeh.
* Match the website's palette so the two artefacts feel like one thing:
  `indigo #818cf8` (primary/total), `emerald #10b981` (train/good), `rose #f43f5e` (validation/bad), `amber #f59e0b` (bias/warning), `sky #38bdf8` (variance).
* Every plot: axis labels with units, a title that states the *finding* rather than the contents ("Validation error bottoms out at degree 5", not "Error vs degree"), and a legend when there is more than one series.
* Never encode meaning by colour alone — vary linestyle or marker too.
* Default figure size around `(7, 4)`; `dpi=110`. Readable on a laptop next to a browser window.
* Annotate the point that matters directly on the plot (the minimum, the crossover, the chosen threshold) with `ax.annotate`.

---

# 12. Technical Requirements

**Dependencies** — pin minimums in `notebooks/requirements.txt`, keep the list short:

```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
scikit-learn>=1.4
jupyterlab>=4.0
```

Optional, guarded by try/except with a graceful fallback and a printed note:
`imbalanced-learn` (SMOTE), `lightgbm`, `xgboost`, `ipywidgets`, `optuna`.

**Reproducibility**

* One `SEED = 7` constant per notebook, passed explicitly everywhere. No global `np.random.seed` reliance.
* Two consecutive Run Alls must produce byte-identical numeric output.
* State the sklearn version in the setup cell — some defaults have moved between releases.

**Runtime**

* Each notebook must Run All in **under 3 minutes** on a laptop CPU.
* No single cell over ~10 seconds. If an experiment genuinely needs more, reduce `n_estimators` or the number of bootstrap draws and say so in a comment.
* No GPU, no network, no multiprocessing surprises.

**Shipping**

* Commit the notebooks **with outputs executed** so they read as a document without being run.
* Clear all `execution_count` noise and ensure no cell errors are stored in the committed output.
* No absolute paths. Everything relative to `notebooks/`.
* `mlprep/` must be importable from the notebook directory without any `sys.path` hacking.

---

# 13. Accuracy Requirements

Same bar as the website, and it matters more here because the notebook makes claims with numbers attached.

* Every stated number must be one the code actually produces. **Never hard-code a result into prose that the cell doesn't compute.** Prefer f-strings that read the live value.
* If a finding is sensitive to the seed, say so and show the spread across several seeds rather than presenting one draw as a law.
* Where the answer depends on context, say so explicitly — "it depends on the cost of false positives versus false negatives" — exactly as the website does.
* Do not oversimplify into inaccuracy. If a demo uses a simplifying assumption (orthonormal features, a specific λ convention), state it in a comment.
* Don't teach deprecated APIs. No `load_boston`, no `normalize=` on estimators, no `sklearn.externals`.
* Where the notebook's number differs from the website's, either reconcile it or explain the difference. They must not silently disagree.

---

# 14. What Not to Do

* **Don't re-teach the website.** No definition dumps, no comparison tables that already exist there. Link to them instead.
* **No 5-row toy DataFrames** to illustrate a concept. Use the real synthetic dataset so results are meaningful.
* **No bare output.** A cell that prints an unexplained array has failed.
* **No `%%time` theatre**, no progress bars on 2-second loops, no ASCII art banners.
* **No mega-cells.** If a cell is 60 lines, it belongs in `mlprep/`.
* **No `warnings.filterwarnings("ignore")` at the top.** Fix the cause, or suppress one specific warning with a comment explaining why.
* **No cell that depends on out-of-order execution.** Run All from a fresh kernel must work every time.
* **Don't chase accuracy.** These are teaching experiments; a 0.81 AUC that demonstrates something is worth more than a 0.89 that demonstrates nothing.

---

# 15. notebooks/README.md

Write it as the entry point:

* What the suite is and how it pairs with the website — one short paragraph.
* Setup: create a venv, `pip install -r requirements.txt`, `jupyter lab`.
* A note that the 📖 links need `npm run dev` running in the repo root, and that the
  notebooks are fully usable without it.
* The four-hour plan as a table: notebook, hour, topics, runtime, matching website sections.
* A "if you only have one hour" path — Notebook 01 §1.3 (leakage), Notebook 02 §2.2–2.3 (thresholds and PR-AUC), Notebook 04 §4.6 (debugging drills).
* A short table of what each notebook proves that the website only asserts. This is the suite's pitch, so make it concrete.
* A troubleshooting note for missing optional dependencies.

---

# 16. Acceptance Checklist

The suite is done when all of these are true:

* [ ] Five notebooks exist, run top-to-bottom from a fresh kernel with no errors, and are committed with outputs.
* [ ] Total runtime across all five is under 12 minutes.
* [ ] Every section links to its matching website anchor and carries a priority label.
* [ ] Every experiment ends with an interpreted result and a 🎤 interview line.
* [ ] The recurring dataset is generated once in `mlprep/data.py` and documented.
* [ ] The overfitting, threshold and PCA experiments reproduce the numbers the website's labs show.
* [ ] At least twelve exercises with working checkers and collapsible solutions.
* [ ] At least eight debugging drills, each quantifying the error its bug caused.
* [ ] Bias–variance is decomposed **empirically**, not illustrated.
* [ ] Random-forest tree correlation is **measured**, not asserted.
* [ ] The PR-AUC-vs-base-rate sweep exists and shows the divergence clearly.
* [ ] The capstone saves a pipeline, reloads it, and asserts identical predictions.
* [ ] Reproducible: two Run Alls give identical numbers.
* [ ] No network access required anywhere.
* [ ] Optional dependencies degrade gracefully with a clear printed message.
* [ ] `notebooks/README.md` and the root `README.md` both updated.
