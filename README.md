Strategic Copilot: Causal Decision Intelligence & AI Executive AssistantStrategic Copilot is an enterprise-grade Causal Decision Intelligence system designed to replace naive observational correlations with structural causal inference and probabilistic decision simulation. By integrating DoWhy graph-based causal validation, EconML Double Machine Learning (DML), and Monte Carlo simulation models with Llama-3.3-70B via the Groq API, Strategic Copilot enables SaaS executives to simulate strategic pricing and marketing intervention outcomes before committing budget.   🌟 Key CapabilitiesObservational & Seasonal Exploratory Analysis: Tracks core SaaS metrics across customer cohorts and historical time horizons while exposing baseline confounding factors.   Structural Causal Modeling & Unbiased Estimation: Uses Directed Acyclic Graphs (DAGs) and Double Machine Learning (DML) to isolate true marginal treatment effects by stripping away confounders like company size.   Probabilistic Decision Intelligence: Runs 5,000-draw Monte Carlo simulations to project Net Financial Impact ($) and Churn Rate (%) across full uncertainty distributions rather than single-point forecasts[cite: 19, 21].AI Strategy Consultant: Automated executive memo generation powered by Groq API and Llama-3.3-70B, converting simulation probability distributions directly into actionable board-ready narratives.   📸 Interactive Application Workflow & Product TourTab 1: Executive Overview & Exploratory Data AnalysisExamines SaaS business health metrics across tracked customer accounts and uncovers observational patterns before causal adjustment.   
Figure 1: Executive Dashboard showing baseline customer portfolio indicators including total account volume, mean MRR, churn rate, and list pricing.      
Figure 2: Historical 12-month trends for Total Monthly Recurring Revenue (MRR) and Monthly Churn Rate highlighting baseline seasonality.      
Figure 3: Interactive variable selector enabling deep exploratory cohort slicing across custom axes.      
Figure 4: Unadjusted scatter plot of Ad Spend vs. MRR displaying a raw observational fit line.      
Figure 5: Full correlation matrix identifying key confounders—such as company size (employees), which strongly co-varies with spend and revenue metrics.   Tab 2: Causal Discovery, DAGs & Estimation DiagnosticsFormulates explicit Structural Causal Models (SCMs) and applies orthogonal machine learning estimators to isolate true treatment effects.   
Figure 6: Directed Acyclic Graph (DAG) explicitly modeling intervention levers, confounders, and primary business target metrics.      
Figure 7: Comparative analysis showing how Naive OLS estimates diverge from Causal DoWhy and Double Machine Learning (EconML) adjustments.      
Figure 8: Rigorous statistical diagnostics table detailing point estimates, 95% confidence bounds, synthetic ground truth benchmarks, and placebo sensitivity tests.   Tab 3: Decision Intelligence & Monte Carlo Risk SimulationTranslates estimated causal effects into probabilistic policy simulations to evaluate risk and outcome dispersion under customized interventions.   
Figure 9: Policy control panel allowing interactive manipulation of monthly ad spend adjustments, price changes, and simulation draw counts.      
Figure 10: Probability density histograms illustrating output distributions for Net Monthly Financial Impact ($) and Projected Monthly Churn (%).      
Figure 11: Statistical outcome breakdown displaying expected mean metrics alongside conservative 5th percentile and optimistic 95th percentile confidence boundaries.   Tab 4: AI Strategy Consultant (LLM Integration)Connects quantitative simulation states directly to an LLM strategy agent for narrative synthesis and decision memo generation.   
Figure 12: Groq API configuration interface for model parameter selection (llama-3.3-70b-versatile) and automated scenario state detection.   🏗️ Repository StructureAll screenshot assets required for application documentation are organized under the assets/ directory:   Plaintextstrategic-copilot/
├── assets/
│   ├── 01_executive_overview.png           #[cite: 22] Tab 1: Executive KPI overview cards
│   ├── 02_mrr_churn_trends.png             #[cite: 18, 22] Tab 1: Total MRR and Churn monthly trends
│   ├── 03_data_explorer_controls.png       #[cite: 17, 22] Tab 1: Interactive X/Y axis selectors
│   ├── 04_naive_scatter.png                #[cite: 16, 22] Tab 1: Raw MRR vs Ad Spend scatter plot
│   ├── 05_correlation_heatmap.png          #[cite: 15, 22] Tab 1: Confounder correlation matrix
│   ├── 06_causal_dag.png                   #[cite: 14, 22] Tab 2: Directed Acyclic Graph (DAG)
│   ├── 07_effect_comparison_barchart.png   #[cite: 13, 22] Tab 2: Naive vs. DoWhy vs. EconML DML
│   ├── 08_causal_diagnostics_table.png     #[cite: 12, 22] Tab 2: Causal effect estimates & refuters
│   ├── 09_decision_simulator_controls.png  #[cite: 11, 22] Tab 3: Policy intervention levers & sliders
│   ├── 10_monte_carlo_distributions.png    #[cite: 10, 19, 22] Tab 3: Net impact & churn histograms
│   ├── 11_simulation_percentiles_table.png #[cite: 9, 21, 22] Tab 3: 5th/95th percentile risk breakdown
│   └── 12_ai_consultant_interface.png      #[cite: 8, 20, 22] Tab 4: Groq API configuration interface
├── data/
│   └── saas_customer_data.csv              # Synthetic customer portfolio dataset
├── src/
│   ├── causal_engine.py                    # DoWhy and EconML DoubleML wrappers
│   ├── simulation.py                       # Monte Carlo sampling routines
│   └── llm_agent.py                        # Groq API prompt engineering & response parsing
├── app.py                                  # Streamlit multi-tab web UI
├── requirements.txt                        # Dependency list
└── README.md                               # System documentation
```[cite: 22]

---

## 📐 Mathematical Framework & Causal Methodology

### 1. Confounder De-biasing via Double Machine Learning (DML)
Standard regression models suffer from omitted variable bias when confounders $X$ (e.g., enterprise scale) influence both treatment $T$ (ad spend/pricing) and outcome $Y$ (MRR/churn). Strategic Copilot uses Robinson's residualization approach:

$$Y - \mathbb{E}[Y\vert{}X] = \theta \cdot (T - \mathbb{E}[T\vert{}X]) + \epsilon$$

Where flexible machine learning estimators partial out the confounding effects of $X$ from both treatment $T$ and outcome $Y$, yielding an unbiased marginal treatment effect estimate $\theta$.

### 2. Probabilistic Monte Carlo Sampling
To account for both parametric estimation uncertainty $\sigma_{\hat{\theta}}$ and underlying binomial churn variance $\sigma_{binomial}$, policy outcomes are sampled across $N = 5,000$ draws:

$$Y_{simulated}^{(i)} \sim \mathcal{N}\left(\hat{\mu}_{\text{impact}}, \sigma_{\hat{\theta}}^2\right) + \text{Binomial}(N_{accounts}, p_{churn}^{(i)})$$

---

## ⚡ Quick Start Guide

### Prerequisites
* Python 3.10 or higher
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/strategic-copilot.git
   cd strategic-copilot
Create and activate a virtual environment:Bashpython -m venv venv
source venv/bin/activate        # On macOS/Linux
# venv\Scripts\activate          # On Windows
Install dependencies:Bashpip install -r requirements.txt
Launch the Streamlit dashboard:Bashstreamlit run app.py
(Optional) Enable AI Strategy Memos:
Provide your Groq API key in the AI Strategy Consultant tab or store it in .env:   BashGROQ_API_KEY="your_groq_api_key_here"
🛠️ Tech StackFrontend & Dashboarding: Streamlit   Causal Inference: DoWhy[cite: 13], EconML (Microsoft Research)[cite: 13]Machine Learning & Modeling: Scikit-Learn, LightGBM, StatsmodelsData Processing & Visualization: Pandas, NumPy, Plotly Express[cite: 18, 19], Graphviz   LLM Orchestration: Groq API (llama-3.3-70b-versatile)