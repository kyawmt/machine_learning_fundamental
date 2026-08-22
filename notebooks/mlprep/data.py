"""
mlprep.data - Synthetic dataset generators for ML interview review experiments.
"""

import math
from typing import Tuple
import numpy as np
import pandas as pd


def _mulberry32(seed: int):
    """Deterministic 32-bit PRNG matching the website's JavaScript implementation."""
    a = int(seed) & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = (a ^ (a >> 15)) * (1 | a)
        t = t & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        return float((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


def _gaussian(rnd, mean: float = 0.0, sd: float = 1.0) -> float:
    """Box-Muller transform matching the website's JavaScript implementation."""
    u = max(rnd(), 1e-9)
    v = rnd()
    return mean + sd * math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def make_churn_panel(
    n_users: int = 4000,
    snapshots_per_user: int = 3,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a synthetic multi-snapshot churn panel dataset with planted learning properties.

    Planted properties:
    - user_id: Repeated rows per user -> Group leakage demo, GroupKFold vs KFold.
    - snapshot_date: Monotone time index ('2025-01-01', '2025-02-01', '2025-03-01')
      -> Temporal split demo, TimeSeriesSplit.
    - tenure_days, monthly_spend, logins_30d: Wildly different scales (10-2000, 15-350, 0-100)
      -> Feature scaling demo.
    - income: MNAR (Missing Not At Random) missingness. Missing probability is ~55% for high earners
      (> $120k) and ~6% for lower earners -> Missingness indicator demo.
    - plan_region: ~200 categorical levels with varying baseline risk
      -> High cardinality, one-hot vs out-of-fold target encoding demo.
    - device: 4 unordered levels ('iOS', 'Android', 'Web', 'Desktop')
      -> One-hot vs ordinal false-ordering trap demo.
    - support_tickets_30d: Window straddles the label timestamp (includes post-event tickets)
      -> Temporal leakage demo.
    - cancellation_reason_code: Only populated for churners (NaN for active users)
      -> Target leakage demo and ablation diagnostics.
    - churned: Binary target with ~6% positive rate -> Class imbalance demo (PR-AUC vs ROC-AUC,
      cost-sensitive thresholds, class weights, SMOTE).
    - Interaction: Monthly spend risk is amplified for low-tenure users, giving non-linear
      tree models an honest performance edge over linear baselines.
    - Honest AUC: A clean model without leaky features achieves ~0.79-0.83 ROC-AUC.
    """
    rng = np.random.RandomState(seed)

    user_ids = [f"USR_{i:04d}" for i in range(1, n_users + 1)]
    snapshot_dates = ["2025-01-01", "2025-02-01", "2025-03-01"][:snapshots_per_user]

    # Generate user-level base attributes
    base_tenure = rng.gamma(shape=2.5, scale=200, size=n_users) + 15  # 15 to ~2000 days
    base_spend = rng.lognormal(mean=4.2, sigma=0.55, size=n_users) + 15  # $15 to ~$350
    base_income = rng.lognormal(mean=11.1, sigma=0.45, size=n_users)  # ~$35k to ~$250k
    base_region_id = rng.randint(1, 201, size=n_users)
    base_region = [f"REG_{r:03d}" for r in base_region_id]
    device_choices = ["iOS", "Android", "Web", "Desktop"]
    base_device = rng.choice(device_choices, p=[0.42, 0.33, 0.15, 0.10], size=n_users)

    # Underlying user churn propensity with non-linear interaction
    # High spend + low tenure = high churn risk; low logins = high churn risk
    region_effects = (base_region_id % 7 - 3) * 0.12
    user_latent_risk = (
        -3.5
        + 0.0035 * base_spend
        - 0.0022 * base_tenure
        + 25.0 / (base_tenure + 40.0) * (base_spend / 60.0)
        + region_effects
    )

    rows = []
    cancellation_reasons = ["Price", "Competitor", "Service", "Relocation"]

    for snap_idx, snap_date in enumerate(snapshot_dates):
        # Time-dependent attributes
        time_offset_days = snap_idx * 30
        snap_tenure = base_tenure + time_offset_days
        snap_spend = np.maximum(15.0, base_spend + rng.normal(0, 4.0, size=n_users))
        # Logins decrease for high-risk users
        expected_logins = np.clip(32.0 - 5.0 * user_latent_risk + rng.normal(0, 6.0, size=n_users), 0, 95)

        # Dynamic logit for this snapshot
        logit = (
            user_latent_risk
            + 0.15 * snap_idx
            - 0.045 * expected_logins
            + rng.normal(0, 0.35, size=n_users)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        churned = (rng.uniform(0, 1, size=n_users) < prob).astype(int)

        # Missingness on income: MNAR (wealthier users withhold income significantly more often)
        mnar_prob = np.where(base_income > 115000, 0.55, 0.06)
        income_missing = rng.uniform(0, 1, size=n_users) < mnar_prob
        snap_income = np.where(income_missing, np.nan, np.round(base_income, -2))

        # Temporal leakage: support tickets measured in a window that straddles the churn event
        # Churners lodge emergency tickets after the decision point
        honest_tickets = np.clip(rng.poisson(lam=np.maximum(0.2, 0.8 + 0.4 * user_latent_risk), size=n_users), 0, 15)
        post_churn_tickets = churned * rng.poisson(lam=3.5, size=n_users)
        leaky_tickets = honest_tickets + post_churn_tickets

        # Target leakage: cancellation reason code is populated ONLY when churned == 1
        reason_codes = [
            rng.choice(cancellation_reasons) if c == 1 else None
            for c in churned
        ]

        for i in range(n_users):
            rows.append({
                "user_id": user_ids[i],
                "snapshot_date": snap_date,
                "tenure_days": float(np.round(snap_tenure[i], 1)),
                "monthly_spend": float(np.round(snap_spend[i], 2)),
                "logins_30d": int(np.round(expected_logins[i])),
                "income": float(snap_income[i]) if not np.isnan(snap_income[i]) else np.nan,
                "plan_region": base_region[i],
                "device": base_device[i],
                "support_tickets_30d": int(leaky_tickets[i]),
                "honest_support_tickets": int(honest_tickets[i]),
                "cancellation_reason_code": reason_codes[i],
                "churned": int(churned[i]),
            })

    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def make_poly_sample(
    n: int = 14,
    noise: float = 0.16,
    seed: int = 344,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the 1-D polynomial regression dataset reproducing the website's Overfitting Lab.

    True function: f(x) = sin(x * pi * 1.65) * 0.85 + x * 0.35
    Returns:
        X_train, y_train, X_val, y_val
    """
    rnd = _mulberry32(seed)

    def true_fn(x: float) -> float:
        return math.sin(x * math.pi * 1.65) * 0.85 + x * 0.35

    xs_train = np.linspace(0.02, 0.98, n)
    xs_val = np.linspace(0.05, 0.95, n)

    y_train = np.array([true_fn(x) + _gaussian(rnd, 0.0, noise) for x in xs_train])
    y_val = np.array([true_fn(x) + _gaussian(rnd, 0.0, noise) for x in xs_val])

    # Convert to column vectors
    X_train = xs_train.reshape(-1, 1)
    X_val = xs_val.reshape(-1, 1)

    return X_train, y_train, X_val, y_val


def make_scored_population(
    n: int = 1000,
    positive_rate: float = 0.03,
    seed: int = 20260822,
) -> pd.DataFrame:
    """
    Generate scored population reproducing the website's Threshold Lab.

    Matches:
    - 970 negatives: squash(gaussian(mean=-3.0, sd=1.6))
    - 30 positives: squash(gaussian(mean=-0.1, sd=1.35))
    - Yields ROC-AUC ≈ 0.94, PR-AUC ≈ 0.41.
    """
    rnd = _mulberry32(seed)

    def squash(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    n_pos = int(round(n * positive_rate))
    n_neg = n - n_pos

    neg_scores = [squash(_gaussian(rnd, -3.0, 1.6)) for _ in range(n_neg)]
    pos_scores = [squash(_gaussian(rnd, -0.1, 1.35)) for _ in range(n_pos)]

    scores = neg_scores + pos_scores
    labels = [0] * n_neg + [1] * n_pos

    df = pd.DataFrame({"score": scores, "y_true": labels})
    return df
