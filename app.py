"""Strategic Copilot - a causal decision intelligence engine (Streamlit UI)."""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.causal_model import EDGES, NODE_POS, estimate_all
from src.data_generator import TRUE_EFFECTS, load_or_generate
from src.simulator import Policy, compute_baseline, simulate

load_dotenv()
st.set_page_config(page_title="Strategic Copilot", page_icon="🧭", layout="wide")

LABELS = {
    "ad_spend->mrr": "Ad spend → MRR ($ per $)",
    "price_tier->mrr": "Price tier → MRR ($ per tier)",
    "price_tier->churn": "Price tier → churn (prob. per tier)",
    "support_tickets_per_user->churn": "Support tickets → churn (prob. per ticket/user)",
}


# ------------------------------------------------------------------ data & models
@st.cache_data(show_spinner="Loading data...")
def get_data() -> pd.DataFrame:
    return load_or_generate()


@st.cache_data(show_spinner="Estimating causal effects (DoWhy + EconML)... ~20s on first run")
def get_effects(df: pd.DataFrame):
    return estimate_all(df)


df = get_data()
effects = get_effects(df)
baseline = compute_baseline(df)

st.title("🧭 Strategic Copilot")
st.caption("A causal decision intelligence engine: simulate interventions with causal effects, not correlations.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview", "🕸️ Causal Graph & Insights",
    "🎛️ Decision Simulator", "🤖 AI Strategy Consultant"])

# ------------------------------------------------------------------ tab 1
with tab1:
    monthly = df.groupby("month").agg(total_mrr=("mrr", "sum"), churn_rate=("churn", "mean"),
                                      avg_ad=("ad_spend", "mean")).reset_index()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{baseline.n_accounts:,}")
    c2.metric("Avg MRR / account", f"${baseline.avg_mrr:,.0f}")
    c3.metric("Monthly churn", f"{baseline.churn_rate:.1%}")
    c4.metric("Avg list price", f"${baseline.avg_price_usd:,.0f}")

    left, right = st.columns(2)
    left.plotly_chart(px.line(monthly, x="month", y="total_mrr", markers=True,
                              title="Total MRR by month"), width="stretch")
    right.plotly_chart(px.line(monthly, x="month", y="churn_rate", markers=True,
                               title="Monthly churn rate"), width="stretch")

    st.subheader("Data explorer")
    numeric = ["ad_spend", "price_tier", "support_tickets_per_user",
               "feature_usage_score", "mrr", "churn", "employees"]
    e1, e2 = st.columns(2)
    x_col = e1.selectbox("X axis", numeric, index=0)
    y_col = e2.selectbox("Y axis", numeric, index=4)
    sample = df.sample(3000, random_state=1)
    st.plotly_chart(px.scatter(sample, x=x_col, y=y_col, opacity=0.4, trendline="ols",
                               title=f"{y_col} vs {x_col} (3,000-row sample; raw correlation only!)"),
                    width="stretch")
    st.plotly_chart(px.imshow(df[numeric].corr().round(2), text_auto=True, aspect="auto",
                              color_continuous_scale="RdBu", zmin=-1, zmax=1,
                              title="Correlation matrix (not causation - see next tab)"),
                    width="stretch")
    with st.expander("Raw data"):
        st.dataframe(df.head(500))

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Assumed causal graph")
    fig = go.Figure()
    for a, b in EDGES:
        (x0, y0), (x1, y1) = NODE_POS[a], NODE_POS[b]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=1.5, arrowcolor="#888",
                           standoff=18, startstandoff=18)
    colors = {"mrr": "#2ca02c", "churn": "#d62728", "company_size": "#7f7f7f",
              "seasonality": "#7f7f7f", "feature_usage": "#7f7f7f"}
    fig.add_trace(go.Scatter(
        x=[p[0] for p in NODE_POS.values()], y=[p[1] for p in NODE_POS.values()],
        mode="markers+text", text=list(NODE_POS.keys()), textposition="top center",
        marker=dict(size=26, color=[colors.get(k, "#1f77b4") for k in NODE_POS]),
        hoverinfo="text"))
    fig.update_layout(height=420, showlegend=False, xaxis=dict(visible=False, range=[-0.7, 5.1]),
                      yaxis=dict(visible=False, range=[-0.6, 3.9]), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue = levers you control · grey = confounders we adjust for · green/red = outcomes. "
               "Company size drives almost everything, which is exactly why raw correlations mislead.")

    st.subheader("Correlation vs. causation")
    rows = []
    for k, e in effects.items():
        rows.append({"effect": LABELS[k], "Naive (correlation)": e.naive,
                     "Causal (DoWhy)": e.dowhy_linear, "Causal (EconML DML)": e.ate})
    comp = pd.DataFrame(rows).melt("effect", var_name="method", value_name="estimate")
    fig2 = px.bar(comp, x="effect", y="estimate", color="method", barmode="group",
                  title="Per-unit effect by method (note the naive bias)")
    fig2.update_xaxes(title=None)
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Effect estimates, uncertainty and diagnostics")
    p0 = baseline.churn_rate
    table = []
    for k, e in effects.items():
        truth = TRUE_EFFECTS[k]
        if k.endswith("churn"):  # ground truth is on the logit scale -> convert to probability slope
            truth = truth * p0 * (1 - p0)
        table.append({"effect": LABELS[k], "naive": round(e.naive, 4), "causal (DML)": round(e.ate, 4),
                      "95% CI": f"[{e.ci_low:.4f}, {e.ci_high:.4f}]",
                      "true value (synthetic)": round(truth, 4),
                      "placebo effect (≈0 is good)": None if e.placebo_effect is None else round(e.placebo_effect, 4),
                      "placebo p-value": None if e.placebo_p_value is None else round(e.placebo_p_value, 2)})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    ad, pm, pc = effects["ad_spend->mrr"], effects["price_tier->mrr"], effects["price_tier->churn"]
    tk = effects["support_tickets_per_user->churn"]
    st.subheader("What actually drives revenue and churn")
    st.markdown(f"""
- **Advertising:** each extra $1 of monthly ad spend per account raises MRR by about **${ad.ate:.2f}**. A naive
  regression claims **${ad.naive:.2f}**, overstating it {ad.naive / ad.ate:.1f}x, because large companies both receive more
  ad spend and pay more anyway.
- **Pricing:** moving up one price tier adds about **${pm.ate:.1f}** of MRR per account, but raises monthly churn by
  **{pc.ate * 100:.2f} percentage points** (95% CI {pc.ci_low * 100:.2f} to {pc.ci_high * 100:.2f}). The naive view
  ({pc.naive * 100:.2f} pp) understates the risk.
- **Support load:** each additional ticket per user raises churn by about **{tk.ate * 100:.2f} pp**, so service quality
  is a retention lever in its own right.
""")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Simulate a business decision")
    s1, s2, s3 = st.columns(3)
    d_ad = s1.slider("Change in monthly marketing budget ($)", -20000, 100000, 20000, 1000)
    d_price = s2.slider("Change in subscription price (%)", -20, 30, 10, 1)
    n_sims = s3.select_slider("Monte Carlo draws", [1000, 5000, 20000], value=5000)

    policy = Policy(delta_ad_spend_total=float(d_ad), delta_price_pct=float(d_price))
    result = simulate(effects, baseline, policy, n_sims=n_sims)
    summ = result.summary()
    st.session_state["sim"] = {"policy": policy, "summary": summ, "prob": result.prob_profitable}

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg MRR / account", f"${summ.loc['avg_mrr_per_account', 'mean']:,.1f}",
              f"{summ.loc['avg_mrr_per_account', 'mean'] - baseline.avg_mrr:+.1f} vs today")
    k2.metric("Monthly churn", f"{summ.loc['churn_rate', 'mean']:.2%}",
              f"{(summ.loc['churn_rate', 'mean'] - baseline.churn_rate) * 100:+.2f} pp", delta_color="inverse")
    k3.metric("Net monthly impact (after ad cost)", f"${summ.loc['net_impact', 'mean']:,.0f}")
    k4.metric("P(net impact > 0)", f"{result.prob_profitable:.0%}")

    def hist(col: str, title: str, fmt: str) -> go.Figure:
        f = px.histogram(result.draws, x=col, nbins=50, title=title)
        for q, dash in ((summ.loc[col, "p5"], "dot"), (summ.loc[col, "mean"], "solid"),
                        (summ.loc[col, "p95"], "dot")):
            f.add_vline(x=q, line_dash=dash, line_color="black")
        f.update_layout(showlegend=False, yaxis_title=None)
        f.update_xaxes(tickformat=fmt)
        return f

    h1, h2 = st.columns(2)
    h1.plotly_chart(hist("net_impact", "Net monthly impact ($): mean and 5th/95th pct", "$,.0f"),
                    width="stretch")
    h2.plotly_chart(hist("churn_rate", "Projected monthly churn: mean and 5th/95th pct", ".2%"),
                    width="stretch")

    out = summ.rename(index={
        "avg_mrr_per_account": "Avg MRR / account ($)", "churn_rate": "Churn rate",
        "retained_mrr": "Total retained MRR ($)", "delta_retained_mrr": "Δ retained MRR vs baseline ($)",
        "net_impact": "Net monthly impact after ad cost ($)"}).rename(
        columns={"p5": "5th pct", "p95": "95th pct"})
    st.dataframe(out.style.format("{:,.4f}", subset=pd.IndexSlice[["Churn rate"], :])
                 .format("{:,.1f}", subset=pd.IndexSlice[out.index.drop("Churn rate"), :]),
                 width="stretch")
    st.caption("Uncertainty comes from (1) the confidence intervals of the causal estimates and "
               "(2) binomial noise in who churns. Effects are assumed linear within the range of historical data; "
               "large moves extrapolate.")

# ------------------------------------------------------------------ tab 4
def offline_memo(sim: dict) -> str:
    s, p = sim["summary"], sim["policy"]
    net = s.loc["net_impact"]
    verdict = "recommend proceeding" if sim["prob"] >= 0.8 else (
        "recommend a smaller pilot first" if sim["prob"] >= 0.5 else "recommend against")
    return (f"**To:** CEO  \n**Re:** Ad spend {p.delta_ad_spend_total:+,.0f} $/mo, price {p.delta_price_pct:+.0f}%\n\n"
            f"Expected net monthly impact is **${net['mean']:,.0f}** (90% interval ${net['p5']:,.0f} to "
            f"${net['p95']:,.0f}), positive in {sim['prob']:.0%} of simulations. Projected churn is "
            f"{s.loc['churn_rate', 'mean']:.2%} versus {baseline.churn_rate:.2%} today. We {verdict}.\n\n"
            "_(Offline template: add a Groq API key to get a full AI-written memo.)_")


def groq_memo(sim: dict, api_key: str, model: str) -> str:
    from groq import Groq

    s, p = sim["summary"], sim["policy"]
    payload = {
        "policy": {"extra_monthly_ad_budget_usd": p.delta_ad_spend_total, "price_change_pct": p.delta_price_pct},
        "baseline": {"accounts": baseline.n_accounts, "avg_mrr": round(baseline.avg_mrr, 2),
                     "monthly_churn": round(baseline.churn_rate, 4)},
        "simulation_mean_p5_p95": json.loads(s.round(4).to_json(orient="index")),
        "prob_net_impact_positive": round(sim["prob"], 3),
        "causal_effects": {k: {"per_unit_effect": round(e.ate, 4), "ci95": [round(e.ci_low, 4), round(e.ci_high, 4)],
                               "naive_correlation": round(e.naive, 4)} for k, e in effects.items()},
    }
    system = ("You are a senior strategy consultant advising the CEO of a SaaS company. Write a concise memo "
              "(under 350 words) with: Recommendation, Expected impact, Key risks, Next steps. Use ONLY the numbers "
              "provided; quantify uncertainty using the 5th-95th percentile range; note that estimates are causal "
              "(confounder-adjusted), not correlations, and that effects are extrapolated linearly.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0.3, max_tokens=800,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": "Simulation results (JSON):\n" + json.dumps(payload, indent=2)}])
    return resp.choices[0].message.content


with tab4:
    st.subheader("AI Strategy Consultant")
    st.write("Reads the simulation currently configured in the **Decision Simulator** tab and writes a CEO memo.")
    key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password",
                        help="Free key at console.groq.com. Can also be set in .env")
    model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    sim = st.session_state.get("sim")
    st.info(f"Scenario: ad budget {sim['policy'].delta_ad_spend_total:+,.0f} $/mo · price "
            f"{sim['policy'].delta_price_pct:+.0f}% · P(profitable) {sim['prob']:.0%}")
    if st.button("Generate CEO memo", type="primary"):
        if not key:
            st.warning("No Groq key found; showing an offline summary instead.")
            st.markdown(offline_memo(sim))
        else:
            try:
                with st.spinner("Consulting Llama..."):
                    st.markdown(groq_memo(sim, key, model))
            except Exception as exc:
                st.error(f"Groq call failed ({exc}). Showing offline summary.")
                st.markdown(offline_memo(sim))
