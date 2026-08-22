export type Priority = "must" | "important" | "good";

export interface SectionMeta {
  /** URL hash + localStorage key */
  id: string;
  /** Full title shown as the section heading */
  title: string;
  /** Compact title for the sidebar */
  nav: string;
  /** 0 = orientation, 1..4 = study-plan hour */
  hour: 0 | 1 | 2 | 3 | 4;
  /** Estimated minutes for a focused pass */
  minutes: number;
  priority: Priority;
  /** One-line description used in the plan + search */
  blurb: string;
  /** Extra keywords so search finds the section by concept name */
  keywords?: string[];
}

export const PRIORITY_LABEL: Record<Priority, string> = {
  must: "Must Know",
  important: "Important",
  good: "Good to Know",
};

export const SECTIONS: SectionMeta[] = [
  {
    id: "study-plan",
    title: "4-Hour Study Plan",
    nav: "4-Hour Study Plan",
    hour: 0,
    minutes: 0,
    priority: "must",
    blurb: "How the four hours are split, and what to do if you only have one.",
    keywords: ["schedule", "plan", "timeline", "progress"],
  },
  {
    id: "problem-types",
    title: "ML Problem Types",
    nav: "ML Problem Types",
    hour: 1,
    minutes: 8,
    priority: "must",
    blurb: "Supervised, unsupervised, semi-supervised, self-supervised. Classification vs regression.",
    keywords: ["supervised", "unsupervised", "semi-supervised", "self-supervised", "classification", "regression", "anomaly detection", "recommendation"],
  },
  {
    id: "splits",
    title: "Train / Validation / Test Split",
    nav: "Train / Val / Test Split",
    hour: 1,
    minutes: 10,
    priority: "must",
    blurb: "What each split is for, common ratios, and stratified vs time-series splitting.",
    keywords: ["holdout", "stratified", "time series split", "ratio", "80/10/10", "test set"],
  },
  {
    id: "data-leakage",
    title: "Data Leakage",
    nav: "Data Leakage",
    hour: 1,
    minutes: 12,
    priority: "must",
    blurb: "The single most common reason an offline model looks great and fails in production.",
    keywords: ["leakage", "fit_transform", "target leakage", "future data", "duplicates"],
  },
  {
    id: "over-underfitting",
    title: "Overfitting and Underfitting",
    nav: "Overfitting & Underfitting",
    hour: 1,
    minutes: 12,
    priority: "must",
    blurb: "Diagnose from the train/validation gap, and know the fix list for each direction.",
    keywords: ["overfit", "underfit", "generalization", "early stopping", "dropout", "augmentation"],
  },
  {
    id: "bias-variance",
    title: "Bias–Variance Tradeoff",
    nav: "Bias–Variance Tradeoff",
    hour: 1,
    minutes: 10,
    priority: "must",
    blurb: "The vocabulary interviewers expect when you explain why a model generalizes badly.",
    keywords: ["bias", "variance", "irreducible error", "noise", "complexity"],
  },
  {
    id: "metrics",
    title: "Model Evaluation Metrics",
    nav: "Evaluation Metrics",
    hour: 2,
    minutes: 22,
    priority: "must",
    blurb: "Confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, and the regression metrics.",
    keywords: ["confusion matrix", "precision", "recall", "f1", "roc", "auc", "pr-auc", "specificity", "mae", "mse", "rmse", "r2", "mape"],
  },
  {
    id: "cross-validation",
    title: "Cross-Validation",
    nav: "Cross-Validation",
    hour: 1,
    minutes: 8,
    priority: "must",
    blurb: "K-Fold and its variants, plus why CV does not replace a held-out test set.",
    keywords: ["k-fold", "stratified", "leave-one-out", "loocv", "nested cv", "time series"],
  },
  {
    id: "feature-engineering",
    title: "Feature Engineering",
    nav: "Feature Engineering",
    hour: 2,
    minutes: 8,
    priority: "important",
    blurb: "Turning raw columns into signal: dates, interactions, logs, binning, aggregations.",
    keywords: ["features", "interaction", "polynomial", "log transform", "binning", "aggregation", "date features"],
  },
  {
    id: "feature-scaling",
    title: "Feature Scaling",
    nav: "Feature Scaling",
    hour: 2,
    minutes: 7,
    priority: "must",
    blurb: "Standardization vs min-max, and exactly which algorithms care.",
    keywords: ["standardization", "z-score", "normalization", "min-max", "robust scaler", "log"],
  },
  {
    id: "missing-data",
    title: "Handling Missing Data",
    nav: "Missing Data",
    hour: 2,
    minutes: 7,
    priority: "important",
    blurb: "Deletion, imputation, missing indicators — and when missingness is itself a feature.",
    keywords: ["imputation", "mean", "median", "mode", "mcar", "mar", "mnar", "indicator"],
  },
  {
    id: "categorical",
    title: "Categorical Variables",
    nav: "Categorical Variables",
    hour: 2,
    minutes: 8,
    priority: "must",
    blurb: "One-hot, ordinal, target encoding, embeddings — and the false-ordering trap.",
    keywords: ["one-hot", "label encoding", "ordinal", "target encoding", "embeddings", "cardinality"],
  },
  {
    id: "class-imbalance",
    title: "Class Imbalance",
    nav: "Class Imbalance",
    hour: 2,
    minutes: 8,
    priority: "must",
    blurb: "Why 99% accuracy can be worthless, and the full toolbox of fixes.",
    keywords: ["imbalance", "smote", "undersampling", "oversampling", "class weights", "threshold"],
  },
  {
    id: "regularization",
    title: "Regularization",
    nav: "Regularization",
    hour: 3,
    minutes: 10,
    priority: "must",
    blurb: "L1 vs L2 vs Elastic Net, what λ does, and how it maps onto bias–variance.",
    keywords: ["l1", "lasso", "l2", "ridge", "elastic net", "lambda", "penalty", "weight decay", "dropout"],
  },
  {
    id: "gradient-descent",
    title: "Gradient Descent",
    nav: "Gradient Descent",
    hour: 3,
    minutes: 10,
    priority: "must",
    blurb: "The update rule, batch vs stochastic vs mini-batch, and learning-rate failure modes.",
    keywords: ["gradient", "learning rate", "sgd", "mini-batch", "momentum", "adam", "convergence"],
  },
  {
    id: "params-hyperparams",
    title: "Hyperparameters vs Parameters",
    nav: "Parameters vs Hyperparameters",
    hour: 3,
    minutes: 7,
    priority: "must",
    blurb: "The distinction plus grid search, random search and Bayesian optimization.",
    keywords: ["grid search", "random search", "bayesian optimization", "tuning", "weights"],
  },
  {
    id: "algorithms",
    title: "Common ML Algorithms",
    nav: "Common ML Algorithms",
    hour: 3,
    minutes: 33,
    priority: "must",
    blurb: "Eight core algorithms as interview cards, plus the big comparison table.",
    keywords: ["linear regression", "logistic regression", "decision tree", "random forest", "gradient boosting", "xgboost", "lightgbm", "knn", "svm", "naive bayes", "kernel"],
  },
  {
    id: "model-selection",
    title: "Model Selection",
    nav: "Model Selection",
    hour: 4,
    minutes: 6,
    priority: "important",
    blurb: "Picking a model from the constraints, not from a leaderboard.",
    keywords: ["baseline", "interpretability", "latency", "no free lunch", "constraints"],
  },
  {
    id: "ensembles",
    title: "Ensemble Methods",
    nav: "Ensemble Methods",
    hour: 4,
    minutes: 7,
    priority: "must",
    blurb: "Bagging, boosting, stacking — and which error component each attacks.",
    keywords: ["bagging", "boosting", "stacking", "voting", "blending", "bootstrap"],
  },
  {
    id: "dimensionality-reduction",
    title: "Dimensionality Reduction",
    nav: "Dimensionality Reduction",
    hour: 4,
    minutes: 6,
    priority: "important",
    blurb: "PCA intuition, explained variance, and when it helps or hurts.",
    keywords: ["pca", "principal component", "explained variance", "t-sne", "umap", "curse of dimensionality"],
  },
  {
    id: "clustering",
    title: "Clustering Fundamentals",
    nav: "Clustering",
    hour: 4,
    minutes: 6,
    priority: "good",
    blurb: "K-Means step by step, choosing K, plus hierarchical and DBSCAN in one line each.",
    keywords: ["k-means", "elbow", "silhouette", "dbscan", "hierarchical", "centroid"],
  },
  {
    id: "pipeline",
    title: "ML Pipeline",
    nav: "ML Pipeline",
    hour: 4,
    minutes: 7,
    priority: "must",
    blurb: "Raw data to monitoring, with a scripted 60–90 second answer for the walk-me-through question.",
    keywords: ["pipeline", "deployment", "monitoring", "drift", "production", "mlops"],
  },
  {
    id: "interview-questions",
    title: "Common Interview Questions",
    nav: "Interview Questions",
    hour: 4,
    minutes: 14,
    priority: "must",
    blurb: "47 grouped Q&As plus 16 scenario drills, all collapsible.",
    keywords: ["questions", "answers", "scenarios", "behavioral", "drill"],
  },
  {
    id: "quiz",
    title: "Rapid-Fire Quiz",
    nav: "Rapid-Fire Quiz",
    hour: 4,
    minutes: 10,
    priority: "must",
    blurb: "82 one-line recall questions with self-scoring. Do this last.",
    keywords: ["quiz", "flashcards", "recall", "score", "practice"],
  },
  {
    id: "cheat-sheet",
    title: "Final Cheat Sheet",
    nav: "Final Cheat Sheet",
    hour: 4,
    minutes: 4,
    priority: "must",
    blurb: "One-page rapid review + the interview readiness checklist.",
    keywords: ["cheat sheet", "summary", "readiness", "checklist", "review"],
  },
];

/** Sections that count toward completion (everything except the plan itself). */
export const TRACKED_SECTIONS = SECTIONS.filter((s) => s.hour !== 0);

export const TOTAL_MINUTES = TRACKED_SECTIONS.reduce((sum, s) => sum + s.minutes, 0);

export const HOURS = [1, 2, 3, 4] as const;

export const HOUR_TITLES: Record<number, { title: string; subtitle: string }> = {
  1: { title: "Hour 1 — Core Concepts", subtitle: "The mental model everything else hangs off" },
  2: { title: "Hour 2 — Metrics & Data Prep", subtitle: "How you measure, and how you feed the model" },
  3: { title: "Hour 3 — Models & Training", subtitle: "The algorithms and the optimization behind them" },
  4: { title: "Hour 4 — Interview Review", subtitle: "Comparison, practice, recall, cheat sheet" },
};

export function sectionsForHour(hour: number): SectionMeta[] {
  return SECTIONS.filter((s) => s.hour === hour);
}

export function minutesForHour(hour: number): number {
  return sectionsForHour(hour).reduce((sum, s) => sum + s.minutes, 0);
}
