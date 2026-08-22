"""
mlprep.plots - Shared matplotlib style and plot helpers matching the website palette.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


# Palette matching the companion website
COLOR_PRIMARY = "#818cf8"    # indigo (primary / total / ensemble)
COLOR_TRAIN = "#10b981"      # emerald (train / good / baseline)
COLOR_VAL = "#f43f5e"        # rose (validation / bad / test)
COLOR_BIAS = "#f59e0b"       # amber (bias / warning / underfitting)
COLOR_VARIANCE = "#38bdf8"   # sky (variance / complexity)
COLOR_NEUTRAL = "#94a3b8"    # slate (unselected / noise)
COLOR_GRID = "#cbd5e1"


def use_style():
    """Apply the clean, laptop-friendly matplotlib style matching the website."""
    plt.rcParams.update({
        "figure.figsize": (7, 4.2),
        "figure.dpi": 110,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#94a3b8",
        "font.family": "sans-serif",
    })


def plot_decision_boundary_2d(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Decision Boundary",
    h: float = 0.02,
):
    """Plot decision boundary for 2D classification problems."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(grid)[:, 1]
        Z = probs.reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors=COLOR_PRIMARY, linewidths=2.0)
    else:
        preds = clf.predict(grid)
        Z = preds.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap="RdYlBu_r", alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.5], colors=COLOR_PRIMARY, linewidths=2.0)

    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="white", linewidths=0.5, s=35
    )
    ax.set_title(title)
    return ax
