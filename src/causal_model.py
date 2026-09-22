"""Causal effect estimation with DoWhy (identification, refutation) and
EconML (double machine learning estimation with confidence intervals)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor

logging.getLogger("dowhy").setLevel(logging.ERROR)

# (treatment, outcome) pairs we estimate, with their back-door adjustment sets.
# Company size, seasonality and product engagement are the confounders.
QUESTIONS: dict[tuple[str, str], list[str]] = {
    ("ad_spend", "mrr"): ["log_employees", "month_sin", "month_cos", "feature_usage_score"],
    ("price_tier", "mrr"): ["log_employees", "feature_usage_score"],
    ("price_tier", "churn"): ["log_employees", "feature_usage_score"],
    ("support_tickets_per_user", "churn"): ["log_employees", "feature_usage_score"],
}

# Assumed causal graph (drives the dashboard visual; the adjustment sets above follow from it).
EDGES = [
    ("company_size", "ad_spend"), ("company_size", "price_tier"),
    ("company_size", "feature_usage"), ("company_size", "support_tickets"),
    ("company_size", "mrr"), ("company_size", "churn"),
    ("seasonality", "ad_spend"), ("seasonality", "mrr"),
    ("feature_usage", "support_tickets"), ("feature_usage", "mrr"), ("feature_usage", "churn"),
    ("ad_spend", "mrr"), ("price_tier", "mrr"), ("price_tier", "churn"),
    ("support_tickets", "churn"),
]
NODE_POS = {
    "company_size": (0, 1.5), "seasonality": (0, 3.2), "feature_usage": (1.4, 0),
    "ad_spend": (2.4, 3.2), "price_tier": (2.4, 1.9), "support_tickets": (2.4, 0.6),
    "mrr": (4.4, 3.0), "churn": (4.4, 0.6),
}


@dataclass
class CausalEffect:
    treatment: str
    outcome: str
    naive: float          # unadjusted slope (what a correlation-based model would claim)
    dowhy_linear: float   # DoWhy back-door linear-regression estimate
    ate: float            # EconML LinearDML effect per one unit of treatment
    se: float
    ci_low: float
    ci_high: float
    placebo_effect: float | None   # DoWhy placebo refuter output (should be ~0)
    placebo_p_value: float | None
    n_obs: int

    @property
    def key(self) -> str:
        return f"{self.treatment}->{self.outcome}"

    def to_dict(self) -> dict:
        return asdict(self) | {"key": self.key}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_employees"] = np.log(out["employees"])
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def _naive_slope(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(t, y)[0, 1] / np.var(t, ddof=1))


def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str],
                    refute: bool = True, seed: int = 0) -> CausalEffect:
    """Estimate the causal effect of `treatment` on `outcome`, adjusting for `confounders`."""
    # 1) DoWhy: encode assumptions, identify the estimand, estimate, then try to refute it.
    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    linear = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    placebo_effect = placebo_p = None
    if refute:
        try:
            ref = model.refute_estimate(estimand, linear, method_name="placebo_treatment_refuter",
                                        placebo_type="permute", num_simulations=20)
            placebo_effect = float(ref.new_effect)
            placebo_p = float(ref.refutation_result["p_value"])
        except Exception:  # refutation is a diagnostic; never block the pipeline
            pass

    # 2) EconML: double machine learning (flexible nuisance models, orthogonalised estimate).
    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        discrete_treatment=False, cv=3, random_state=seed,
    )
    dml.fit(df[outcome].to_numpy(), df[treatment].to_numpy(), X=None, W=df[confounders].to_numpy())
    inf = dml.ate_inference(X=None)
    ate = float(np.squeeze(inf.mean_point))
    se = float(np.squeeze(inf.stderr_mean))
    lo, hi = (float(np.squeeze(v)) for v in inf.conf_int_mean(alpha=0.05))

    return CausalEffect(
        treatment=treatment, outcome=outcome,
        naive=_naive_slope(df[treatment].to_numpy(float), df[outcome].to_numpy(float)),
        dowhy_linear=float(linear.value), ate=ate, se=se, ci_low=lo, ci_high=hi,
        placebo_effect=placebo_effect, placebo_p_value=placebo_p, n_obs=len(df),
    )


def estimate_all(df: pd.DataFrame, sample: int = 12000, refute: bool = True,
                 seed: int = 0) -> dict[str, CausalEffect]:
    """Estimate every causal question in QUESTIONS on a random sample of the panel."""
    data = prepare_features(df)
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)
    data = data.reset_index(drop=True)
    results = {}
    for (t, y), confounders in QUESTIONS.items():
        eff = estimate_effect(data, t, y, confounders, refute=refute, seed=seed)
        results[eff.key] = eff
    return results


def compute_treatment_effect(effects: dict[str, CausalEffect], key: str, delta: float) -> dict:
    """Change in the outcome from moving the treatment by `delta` units (point estimate + 95% CI)."""
    e = effects[key]
    return {"effect": e.ate * delta, "ci_low": e.ci_low * delta, "ci_high": e.ci_high * delta}


if __name__ == "__main__":
    from src.data_generator import load_or_generate, TRUE_EFFECTS

    res = estimate_all(load_or_generate())
    for k, e in res.items():
        print(f"{k:34s} naive={e.naive:8.3f} dowhy={e.dowhy_linear:8.3f} "
              f"dml={e.ate:8.3f} [{e.ci_low:.3f}, {e.ci_high:.3f}] "
              f"placebo={e.placebo_effect} truth(logit/$)={TRUE_EFFECTS[k]}")
