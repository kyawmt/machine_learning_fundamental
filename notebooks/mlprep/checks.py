"""
mlprep.checks - Automated exercise assertion helpers with actionable feedback.
"""

from typing import Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from mlprep.data import make_churn_panel, make_scored_population


def check_1_2(split_fn: Callable):
    """Check exercise 1.2: split_honestly(df) respecting both time and user grouping."""
    df = make_churn_panel(n_users=600, snapshots_per_user=3, seed=42)
    try:
        res = split_fn(df)
        if not isinstance(res, (tuple, list)) or len(res) != 2:
            print("❌ Expected split_honestly to return a tuple of (train_df, test_df).")
            return
        train_df, test_df = res
        if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
            print("❌ Both train_df and test_df must be pandas DataFrames.")
            return
        if len(train_df) == 0 or len(test_df) == 0:
            print("❌ Neither train_df nor test_df can be empty.")
            return

        # Check user leakage
        train_users = set(train_df["user_id"].unique())
        test_users = set(test_df["user_id"].unique())
        overlap = train_users.intersection(test_users)
        if len(overlap) > 0:
            print(f"❌ User leakage detected: {len(overlap)} user_id(s) appear in both train and test sets!")
            return

        # Check temporal order
        train_max_date = train_df["snapshot_date"].max()
        test_min_date = test_df["snapshot_date"].min()
        if train_max_date > test_min_date:
            print(f"❌ Temporal leakage: latest train date ({train_max_date.date()}) is after earliest test date ({test_min_date.date()})!")
            return

        print(f"✅ Correct! Split respects both time ordering (Train <= {train_max_date.date()}, Test >= {test_min_date.date()}) and user isolation (0 user overlap).")
    except Exception as e:
        print(f"❌ Error while running split_honestly: {e}")


def check_1_3(audit_fn: Callable):
    """Check exercise 1.3: audit_leakage(df, target, time_col, group_col)."""
    df = make_churn_panel(n_users=500, snapshots_per_user=3, seed=42)
    try:
        report = audit_fn(df, target="churned", time_col="snapshot_date", group_col="user_id")
        if not isinstance(report, pd.DataFrame):
            print("❌ Expected audit_leakage to return a pandas DataFrame.")
            return
        if "cancellation_reason_code" not in report.index and "cancellation_reason_code" not in report.get("feature", []):
            print("❌ Feature 'cancellation_reason_code' must be in the audit report.")
            return
        if "cancellation_reason_code" in report.index and "suspicious_leakage" in report.columns:
            if not bool(report.loc["cancellation_reason_code", "suspicious_leakage"]):
                print("❌ 'cancellation_reason_code' should be flagged as suspicious leakage.")
                return
        print("✅ Correct! Leakage audit successfully created and flagged target/temporal risks.")
    except Exception as e:
        print(f"❌ Error while running audit_leakage: {e}")


def check_1_4(diagnose_fn: Callable):
    """Check exercise 1.4: diagnose_fit(train_score, val_score)."""
    try:
        d1 = diagnose_fn(0.99, 0.65).lower()
        d2 = diagnose_fn(0.60, 0.58).lower()
        d3 = diagnose_fn(0.85, 0.84).lower()

        if "overfit" not in d1 and "variance" not in d1:
            print("❌ (Train 0.99, Val 0.65) is a classic case of Overfitting / High Variance.")
            return
        if "underfit" not in d2 and "bias" not in d2:
            print("❌ (Train 0.60, Val 0.58) is a classic case of Underfitting / High Bias.")
            return
        if "well" not in d3 and "good" not in d3 and "balanced" not in d3:
            print("❌ (Train 0.85, Val 0.84) indicates a Well-fit / balanced model.")
            return
        print("✅ Correct! All 3 diagnostics and interventions correctly identified.")
    except Exception as e:
        print(f"❌ Error while running diagnose_fit: {e}")


def check_2_1(precision_fn, recall_fn, specificity_fn, f1_fn):
    """Check exercise 2.1: confusion matrix metrics from scratch."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0])

    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    try:
        p = precision_fn(tp, fp, fn, tn)
        r = recall_fn(tp, fp, fn, tn)
        s = specificity_fn(tp, fp, fn, tn)
        f = f1_fn(tp, fp, fn, tn)

        exp_p = precision_score(y_true, y_pred)
        exp_r = recall_score(y_true, y_pred)
        exp_s = tn / (tn + fp)
        exp_f = f1_score(y_true, y_pred)

        if not np.isclose(p, exp_p):
            print(f"❌ Precision mismatch: got {p:.4f}, expected {exp_p:.4f}")
            return
        if not np.isclose(r, exp_r):
            print(f"❌ Recall mismatch: got {r:.4f}, expected {exp_r:.4f}")
            return
        if not np.isclose(s, exp_s):
            print(f"❌ Specificity mismatch: got {s:.4f}, expected {exp_s:.4f}")
            return
        if not np.isclose(f, exp_f):
            print(f"❌ F1 mismatch: got {f:.4f}, expected {exp_f:.4f}")
            return

        print(f"✅ Correct! Precision ({p:.3f}), Recall ({r:.3f}), Specificity ({s:.3f}), and F1 ({f:.3f}) match scikit-learn exactly.")
    except Exception as e:
        print(f"❌ Error evaluating metric functions: {e}")


def check_2_2(find_best_threshold_fn: Callable):
    """Check exercise 2.2: cost-optimal threshold."""
    df = make_scored_population(n=1000, positive_rate=0.03, seed=20260822)
    try:
        best_t, min_cost = find_best_threshold_fn(
            y_true=df["y_true"].values,
            y_scores=df["score"].values,
            cost_fp=50.0,
            cost_fn=200.0,
        )
        if not (0.10 <= best_t <= 0.40):
            print(f"❌ Optimal threshold {best_t:.3f} is outside the expected range [0.10, 0.40] for cost_fp=50, cost_fn=200.")
            return
        print(f"✅ Correct! Optimal threshold found: {best_t:.2f} with total cost ${min_cost:,.2f}.")
    except Exception as e:
        print(f"❌ Error in find_best_threshold: {e}")


def check_3_1(momentum_gd_fn: Callable):
    """Check exercise 3.1: Momentum Gradient Descent."""
    try:
        # Run test on 1D quadratic J(w) = 0.5 * (w - 3)^2
        grad_fn = lambda w: w - 3.0
        w_final, history = momentum_gd_fn(grad_fn, w_start=-5.0, lr=0.1, beta=0.8, n_steps=60)
        if not np.isclose(w_final, 3.0, atol=0.05):
            print(f"❌ Momentum GD did not converge to optimum w=3.0 (reached {w_final:.4f}).")
            return
        print(f"✅ Correct! Momentum GD converged to optimum w={w_final:.4f} in {len(history)} steps.")
    except Exception as e:
        print(f"❌ Error running momentum_gd: {e}")


def check_3_2(log_loss_grad_fn: Callable):
    """Check exercise 3.2: Logistic regression analytical gradient."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3)
    y = rng.randint(0, 2, size=20)
    w = rng.randn(3)

    try:
        grad = log_loss_grad_fn(X, y, w)
        if grad.shape != w.shape:
            print(f"❌ Expected gradient shape {w.shape}, got {grad.shape}.")
            return
        # Finite-difference check
        eps = 1e-6
        grad_num = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += eps
            p_plus = 1.0 / (1.0 + np.exp(-X @ w_plus))
            loss_plus = -np.mean(y * np.log(p_plus + 1e-12) + (1 - y) * np.log(1 - p_plus + 1e-12))

            w_minus = w.copy()
            w_minus[i] -= eps
            p_minus = 1.0 / (1.0 + np.exp(-X @ w_minus))
            loss_minus = -np.mean(y * np.log(p_minus + 1e-12) + (1 - y) * np.log(1 - p_minus + 1e-12))

            grad_num[i] = (loss_plus - loss_minus) / (2 * eps)

        if not np.allclose(grad, grad_num, atol=1e-4):
            print(f"❌ Analytical gradient {grad} differs from numerical gradient {grad_num}!")
            return
        print("✅ Correct! Analytical gradient matches finite-difference gradient.")
    except Exception as e:
        print(f"❌ Error in log_loss_grad: {e}")


def check_4_2(ratio_calc_fn: Callable):
    """Check exercise 4.2: Bagging variance reduction ratio."""
    try:
        r = ratio_calc_fn(var_single=0.48, var_ensemble=0.06)
        if not np.isclose(r, 0.125):
            print(f"❌ Expected ratio 0.125, got {r}")
            return
        print(f"✅ Correct! Variance reduction ratio is {r:.3f} (87.5% variance reduction).")
    except Exception as e:
        print(f"❌ Error: {e}")


def check_4_4(kmeans_fn: Callable):
    """Check exercise 4.4: K-Means from scratch."""
    rng = np.random.RandomState(42)
    # 3 distinct 2D clusters
    c1 = rng.randn(30, 2) + [0, 5]
    c2 = rng.randn(30, 2) + [5, 0]
    c3 = rng.randn(30, 2) + [5, 5]
    X = np.vstack([c1, c2, c3])

    try:
        centroids, labels, inertia = kmeans_fn(X, k=3, max_iter=30, seed=42)
        if centroids.shape != (3, 2):
            print(f"❌ Expected centroids shape (3, 2), got {centroids.shape}.")
            return
        if len(labels) != len(X):
            print(f"❌ Expected {len(X)} labels, got {len(labels)}.")
            return
        print(f"✅ Correct! K-Means converged with final inertia = {inertia:.2f}.")
    except Exception as e:
        print(f"❌ Error running kmeans: {e}")
