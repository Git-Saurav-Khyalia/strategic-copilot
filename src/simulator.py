"""Monte Carlo simulator: turns causal effect estimates into probabilistic projections.

Two sources of uncertainty are propagated on every draw:
  1. Parameter uncertainty - each causal effect is drawn from N(ATE, SE) (from EconML DML).
  2. Outcome noise         - the number of churned accounts is Binomial(N, churn probability).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.causal_model import CausalEffect
from src.data_generator import PRICE_PER_TIER_USD


@dataclass
class Policy:
    delta_ad_spend_total: float = 0.0   # extra $ / month of ad spend across all accounts
    delta_price_pct: float = 0.0        # % change in list price


@dataclass
class Baseline:
    n_accounts: int
    avg_mrr: float
    churn_rate: float
    avg_ad_spend: float
    avg_price_usd: float


def compute_baseline(df: pd.DataFrame, last_n_months: int = 3) -> Baseline:
    """Current state of the business, averaged over the most recent months."""
    recent = df[df["month"] > df["month"].max() - last_n_months]
    return Baseline(
        n_accounts=int(df["account_id"].nunique()),
        avg_mrr=float(recent["mrr"].mean()),
        churn_rate=float(recent["churn"].mean()),
        avg_ad_spend=float(recent["ad_spend"].mean()),
        avg_price_usd=float(recent["price_usd"].mean()),
    )


@dataclass
class SimulationResult:
    draws: pd.DataFrame
    baseline: Baseline
    policy: Policy

    def summary(self) -> pd.DataFrame:
        rows = {}
        for col in self.draws.columns:
            x = self.draws[col]
            rows[col] = {"mean": x.mean(), "p5": x.quantile(0.05), "p95": x.quantile(0.95)}
        return pd.DataFrame(rows).T

    @property
    def prob_profitable(self) -> float:
        return float((self.draws["net_impact"] > 0).mean())


def simulate(effects: dict[str, CausalEffect], baseline: Baseline, policy: Policy,
             n_sims: int = 5000, seed: int = 7) -> SimulationResult:
    """Project MRR and churn under `policy`, compared with the do-nothing baseline."""
    rng = np.random.default_rng(seed)
    n = baseline.n_accounts

    d_ad = policy.delta_ad_spend_total / n                                  # $ per account
    d_tier = policy.delta_price_pct / 100 * baseline.avg_price_usd / PRICE_PER_TIER_USD

    def draw(key: str) -> np.ndarray:
        e = effects[key]
        return rng.normal(e.ate, e.se, n_sims)

    tau_ad_mrr, tau_price_mrr, tau_price_churn = (
        draw("ad_spend->mrr"), draw("price_tier->mrr"), draw("price_tier->churn"))

    avg_mrr = baseline.avg_mrr + tau_ad_mrr * d_ad + tau_price_mrr * d_tier
    p_churn = np.clip(baseline.churn_rate + tau_price_churn * d_tier, 0.0, 1.0)

    churners = rng.binomial(n, p_churn)
    churners_base = rng.binomial(n, baseline.churn_rate, n_sims)

    retained = (n - churners) * avg_mrr
    retained_base = (n - churners_base) * baseline.avg_mrr
    delta_retained = retained - retained_base

    draws = pd.DataFrame({
        "avg_mrr_per_account": avg_mrr,
        "churn_rate": churners / n,
        "retained_mrr": retained,
        "delta_retained_mrr": delta_retained,
        "net_impact": delta_retained - policy.delta_ad_spend_total,   # monthly, after ad cost
    })
    return SimulationResult(draws=draws, baseline=baseline, policy=policy)
