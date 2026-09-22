"""Synthetic SaaS data with a *known* causal structure.

Because we write the data-generating process ourselves, we know the true
causal effects and can check that the causal model recovers them (and that
naive correlations do not).

True structural equations (per account-month):

    size_z            ~ N(0, 1)                       # confounder (company size)
    usage_z           = 0.5*size_z + noise            # product engagement
    ad_spend          = 100 + 25*size_z + 20*season + noise
    price_tier        = round(2.6 + 0.6*size_z + noise), clipped to 1..5
    support_tickets   ~ Gamma(mean = exp(0.2*size_z - 0.15*usage_z))
    mrr               = 60 + 22*size_z + 0.40*ad_spend + 18*price_tier
                        + 8*usage_z + 6*season + noise
    P(churn)          = sigmoid(-3.0 + 0.35*(tier-2.6) + 0.30*(tickets-1)
                                - 0.40*usage_z - 0.10*size_z)

Company size drives ad spend, price tier, tickets and revenue at once, so
naive correlations are biased; only a model that adjusts for it recovers the
true effects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_BASE_USD = 20.0
PRICE_PER_TIER_USD = 15.0

# Ground truth used to generate the data (churn effects are on the logit scale).
TRUE_EFFECTS = {
    "ad_spend->mrr": 0.40,                      # $ MRR per $ of monthly ad spend
    "price_tier->mrr": 18.0,                    # $ MRR per price tier
    "price_tier->churn": 0.35,                  # logit units per tier
    "support_tickets_per_user->churn": 0.30,    # logit units per ticket/user
}

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "saas_historical_data.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_saas_data(n_accounts: int = 5000, n_months: int = 12, seed: int = 42) -> pd.DataFrame:
    """Return an account-month panel (n_accounts * n_months rows)."""
    rng = np.random.default_rng(seed)
    shape = (n_accounts, n_months)

    size_z = np.repeat(rng.normal(0, 1, (n_accounts, 1)), n_months, axis=1)
    usage_base = np.repeat(rng.normal(0, 0.87, (n_accounts, 1)), n_months, axis=1)
    usage_z = 0.5 * size_z + 0.8 * usage_base + rng.normal(0, 0.5, shape)

    months = np.tile(np.arange(1, n_months + 1), (n_accounts, 1))
    season = np.sin(2 * np.pi * months / 12)

    ad_spend = np.clip(100 + 25 * size_z + 20 * season + rng.normal(0, 30, shape), 0, None)
    tier = np.clip(np.round(2.6 + 0.6 * size_z + rng.normal(0, 0.8, shape)), 1, 5)
    tickets = rng.gamma(2.0, np.exp(0.2 * size_z - 0.15 * usage_z) / 2.0)

    mrr = (60 + 22 * size_z + 0.40 * ad_spend + 18 * tier + 8 * usage_z
           + 6 * season + rng.normal(0, 15, shape))
    mrr = np.clip(mrr, 5, None)

    logit = (-3.0 + 0.35 * (tier - 2.6) + 0.30 * (tickets - 1.0)
             - 0.40 * usage_z - 0.10 * size_z)
    churn = (rng.uniform(size=shape) < _sigmoid(logit)).astype(int)

    return pd.DataFrame({
        "account_id": np.repeat(np.arange(n_accounts), n_months),
        "month": months.ravel(),
        "employees": np.round(np.exp(3.5 + size_z.ravel())).astype(int),
        "ad_spend": ad_spend.ravel().round(2),
        "price_tier": tier.ravel().astype(int),
        "price_usd": (PRICE_BASE_USD + PRICE_PER_TIER_USD * tier.ravel()).round(2),
        "support_tickets_per_user": tickets.ravel().round(3),
        "feature_usage_score": np.clip(55 + 12 * usage_z.ravel(), 0, 100).round(1),
        "mrr": mrr.ravel().round(2),
        "churn": churn.ravel(),
    })


def load_or_generate(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the CSV, generating it first if it does not exist yet."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_saas_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate_saas_data()
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DEFAULT_PATH, index=False)
    print(f"Wrote {len(data):,} rows to {DEFAULT_PATH}")
    print(data.describe().T[["mean", "std", "min", "max"]].round(2))
