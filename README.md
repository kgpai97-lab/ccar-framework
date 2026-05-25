# CCAR-Style Capital Forecasting & Stress Testing Framework

An end-to-end, regulator-style simulation of the Federal Reserve's **Comprehensive Capital Analysis and Review (CCAR)** program. The framework ingests Federal Reserve supervisory macroeconomic scenarios and a representative U.S. G-SIB's portfolio disclosures, projects portfolio-level credit losses and pre-provision revenue over a multi-year horizon, and translates them into a quarterly forward path for the bank's **Common Equity Tier 1 (CET1) capital ratio** under both the 2026 Supervisory Baseline and Severely Adverse scenarios.

The codebase is built to mirror the operational architecture of a bank capital-planning team: a macro factor library, four independent econometric component models (retail PD, retail LGD, wholesale LGD, PPNR), and a capital projection engine anchored to the bank's most recent regulatory filings.

---

## Quantitative Executive Summary

| Layer | Method | Output |
|---|---|---|
| Macro factor library | FRED pull → quarterly resample → stationarity transforms | 30+ aligned macro factors, 2000Q1–2029Q1 |
| Retail PD | Engle-Granger two-step Error Correction Model on logit(DR) | Quarterly retail PD path under both scenarios |
| Retail LGD | Logit-OLS with LASSO-CV / stepwise feature selection on a 60+ candidate pool | Quarterly retail LGD path (final shipped model uses Fed 90% prior for regulatory credibility) |
| Wholesale LGD | OLS with Newey-West HAC standard errors, narrative model build M0 → M5 | Quarterly wholesale LGD path |
| PPNR | Log-linear AR(1) on bank assets and 10Y Treasury yield | Quarterly PPNR path under both scenarios |
| Capital projection | Basel expected-loss accounting + net-income waterfall + RWA evolution | Quarterly CET1 ratio vs. 4.5% regulatory minimum |

---

## Core Functionality

### 1. Macro Scenario Mapping
- Pulls 21 FRED series — unemployment, real and nominal GDP, BBB / HY / AAA / AA OAS spreads, VIX, NFCI, STLFSI4, 10Y and 3M Treasury yields, prime rate, CPI, real disposable income, HPI, NASDAQ, industrial production, capacity utilization — and resamples daily / weekly / monthly frequencies to quarter-end means.
- Constructs the stationarity transforms used by every downstream model: QoQ annualized growth for GDP and RDI, YoY growth for HPI / CPI / corporate profits / industrial production, first-differences for rates and spreads, and the 10Y–3M term-spread.
- Maps the **Fed's 2026 Supervisory Baseline and Severely Adverse domestic scenario tables** into the same column schema as the historical macro panel so that the same model coefficients can be applied to both historical fit and forward path.

### 2. Portfolio Credit Loss Modeling

The framework decomposes credit losses into four independent econometric specifications — one per (portfolio × parameter) cell — rather than fitting a single joint loss model. This mirrors the modular structure used by bank capital-planning teams, where PD, LGD, and EAD are typically owned by separate model groups.

#### 2a. Retail PD — Engle-Granger Two-Step Error Correction Model

**Target.** The Fed H.8 retail delinquency rate `DR_t`, logit-transformed to map the bounded [0, 1] series onto the real line for linear modeling:

`logit_DR_t = ln(DR_t / (1 − DR_t))`

with an ε = 1e-6 boundary clip to guard against log(0) at extremes.

**Why ECM, not direct OLS.** An initial level-on-level OLS on `logit_DR` produced a near-unit-root coefficient on the lagged dependent variable — economically implausible and a textbook sign of a spurious regression on non-stationary credit series. The notebook explicitly rejects that specification and switches to Engle-Granger, which separates the **long-run cointegrating equilibrium** from the **short-run dynamics** that drive quarterly fluctuations.

**Stage 1 — Long-run equilibrium regression** (in levels):

```
logit_DR_t = α + β₁·Unemployment_t + β₂·Prime_Rate_t + β₃·CPI_t + u_t
```

The residual `u_t` is the disequilibrium error — the deviation of the current logit-DR from its macro-implied long-run level. Estimated coefficients: α = −3.34, β_unemp = 0.105, β_prime = 0.135, β_cpi = −0.006.

**Stage 2 — Short-run dynamics** (in differences):

```
Δlogit_DR_t = γ₀ + γ₁·ΔUnemp_{t-1} + γ₂·ΔUnemp_{t-2}
            + γ₃·ΔPrime_t + γ₄·ΔBBB_Spread_t
            + γ₅·Δlogit_DR_{t-1}
            + λ·u_{t-1} + ε_t
```

The **error-correction coefficient λ = −0.0211** is the key parameter: when the prior quarter's logit-DR sat above its macro-implied equilibrium, the next quarter's change is pulled back down by ~2.1% of the gap. Lagged differences in unemployment (1 and 2 quarter lags) capture the delayed labor-market transmission into consumer credit deterioration; BBB spread differences capture the wholesale credit-conditions channel that bleeds into retail behavior.

**Forecasting mechanics.** A recursive loop carries forward `logit_DR_{t-1}`, `Δlogit_DR_{t-1}`, `ΔUnemp_{t-1}`, `ΔUnemp_{t-2}` and the macro level lags, reconstructing the long-run equilibrium and the EC term at each step, applying the short-run equation, and inverse-logit-transforming back to a PD path.

**Diagnostics.** VIF on the short-run regressors and in-sample fitted-vs-actual overlay. A COVID-period dummy was evaluated during specification search and rejected — the error-correction term absorbed the 2020–2021 residuals without a one-off indicator, preserving model parsimony.

#### 2b. Retail LGD — Logit-OLS with Regularized Feature Selection

**Target.** Recovery-derived retail LGD, logit-transformed identically to the PD target to keep predictions in [0, 1].

**Feature engineering — the candidate pool.** Fifteen "core" macro variables are each transformed in three ways depending on their economic character:
- **Percent-change variables** (stock market, real GDP, RDI, CPI) — QoQ pct_change.
- **Absolute-change variables** (prime, 10Y, 3M) — first difference.
- **Level variables** (unemployment, VIX, HY/BBB spreads, financial stress index, HPI YoY) — kept in levels because they are already stationary stress proxies.

Each transformed series is then **lagged 1 through 4 quarters**, producing a candidate pool of ~75 features. This is the standard "macro shotgun" approach used in bank LGD modeling — let the data tell you which lag of which variable matters, rather than picking ex ante.

**Two parallel selection methods are run and compared:**

1. **Stepwise forward/backward selection** with p-value thresholds (in = 0.01, out = 0.05). At each step, the variable with the lowest p-value below the entry threshold is added; the variable with the highest p-value above the exit threshold is dropped. Iterates until no further changes.
2. **LASSO with 5-fold cross-validation** on standardized features (`StandardScaler` is required because LASSO penalizes coefficient magnitudes). Cross-validation selects the regularization parameter α; the non-zero coefficients identify the surviving features. An OLS refit on those features then recovers proper p-values and HC standard errors that LASSO does not natively provide.

The selection logic is wrapped in a reusable `LGDLassoModel` class with `prepare_data`, `select_features`, `fit_final_model`, `predict`, and `evaluate` methods — production-style scaffolding rather than ad-hoc scripting.

**Final shipped specification** (manual override after reviewing data-driven outputs):

```
logit_LGD_t = α + β₁·Unemployment_t + β₂·ΔPrime_Rate_{t-2} + ε_t
```

**Why the manual override.** Both the stepwise and LASSO pipelines flagged variables that explained variance but produced **counterintuitive signs over the 2022–2023 recovery window** — particularly an under-sized coefficient on unemployment, which is theoretically a primary LGD driver. Rather than ship a model whose signs would fail regulator scrutiny, the project explicitly trades statistical fit for economic interpretability and adopts the Fed's **prescribed 90% retail LGD assumption** in the final capital projection, treating the regression as a research artifact rather than the production input.

#### 2c. Wholesale LGD — OLS with HAC Inference and Narrative Model Build

**Target construction.** Wholesale LGD is engineered from Fed H.8 panels as:

```
Wholesale_LGD_t = Wholesale_NCO_t / Wholesale_DR_t
```

where NCO is net charge-offs and DR is the delinquency-rate proxy. Quarters with `DR = 0` produce undefined LGD and are dropped.

**Feature pool.** A 36-variable macro candidate set (levels, growth rates, first-differences, term-spread, change-in-rate series) is each lagged 1 to 4 quarters, producing ~150 candidate columns. The LGD's own first four lags are also included.

**Multicollinearity pruning pipeline** — three sequential filters:

1. **Top-30 by absolute correlation with the target** — fast univariate ranking.
2. **Dedup by base name** — a custom `base_name()` regex strips `_LagN` suffixes so the same variable doesn't dominate the shortlist at multiple lags.
3. **Pairwise correlation cap at |ρ| > 0.95** via the `drop_high_corr_features` function, which accepts a `prefer_keep` set of family priorities — e.g., keep `3M_Change` over `Prime_Change` (cleaner policy-rate signal) and `BBB_Spread` over `AA_Spread` / `HY_Spread` (mid-grade corporate credit proxy).

**Narrative model build (M0 → M5).** Thirteen nested specifications are compared on a common sample. Each step adds one economically motivated variable and is scored on:
- **Adjusted R²** (explanatory power, penalized for parameter count)
- **AIC and BIC** (information-criterion model selection)
- **Δ vs. previous spec** for each of the above
- **Economic sign check** against a hand-coded `expected_sign` dictionary that pins down the prior for every candidate — e.g., HPI YoY should have a negative sign (rising house prices improve recoveries → lower LGD), BBB spread should be positive (widening corporate spreads predict deteriorating recoveries → higher LGD). Coefficients whose actual sign contradicts the prior are flagged `WRONG` in the comparison table.

**Final specification:**

```
Wholesale_LGD_t = α + β₁·Wholesale_LGD_{t-1}
                + β₂·HPI_YoY_{t-1}
                + β₃·Δ3M_Treasury_t
                + β₄·VIX_t + ε_t
```

The lag-1 term gives the model the persistence that wholesale loss-given-default empirically exhibits; HPI YoY captures the collateral-value channel; the 3M Treasury change captures the policy-tightening transmission into corporate stress; VIX captures broader market-stress conditions.

**Residual diagnostics — the full stack:**
- **Durbin-Watson** — tests for AR(1) residual autocorrelation; values near 2 indicate clean residuals.
- **Ljung-Box at lags 4 and 8** — joint test for residual autocorrelation at one-year and two-year horizons.
- **Newey-West HAC standard errors with `maxlags = 4`** — robust inference under any residual autocorrelation the diagnostics may have flagged; this is the standard approach in quarterly time-series finance models.
- **VIF on the final regressors** — confirms no remaining multicollinearity post-pruning.

**Forecasting mechanics.** A `run_lgd_scenario()` function is parameterized over scenario CSV path, scenario name, and a "bridge quarter" (2025Q4) that splices the actual 2025Q3 historical LGD onto the forecast horizon, then recursively projects 2026Q1 through 2029Q1.

#### 2d. Wholesale PD

The wholesale PD path is sourced as a **pre-computed quarterly series** for each Fed scenario (`PD forecast/Probability Default - Baseline.csv` and `Probability Default - Severely Adverse.csv`) from a parallel modeling workstream documented separately. The CET1 engine consumes this series as a fixed input. The notebook in the `PD forecast/` directory is a plotting and QA harness, not a fitting routine.

### 3. Balance Sheet & Capital Forecasting

#### 3a. PPNR — Log-Linear Autoregression

**Target.** Pre-Provision Net Revenue (PPNR), extracted from quarterly 10-K filings spanning 2010Q1 to 2025Q2. PPNR is the bank's earnings power before loan loss provisions — net interest income plus non-interest income, minus non-interest expense.

**Why log-linear.** PPNR exhibits strong exponential drift and proportional volatility — a doubling of the bank's balance sheet roughly doubles both the level and the absolute quarterly variation in PPNR. A log transform stabilizes the variance and makes the coefficients elasticities (a 1-unit move in the regressor produces a percent change in PPNR).

**Specification:**

```
log(PPNR_t) = α + β₁·log(PPNR_{t-1})
            + β₂·(Total_Assets_t / 1e6)
            + β₃·10Y_Treasury_Yield_t + ε_t
```

- `log(PPNR_{t-1})` — the AR(1) term captures revenue persistence; a coefficient < 1 ensures mean reversion.
- `Total_Assets_t / 1e6` — scaled to keep coefficients numerically interpretable; the natural exogenous driver of revenue capacity.
- `10Y_Treasury_Yield_t` — captures the term-structure channel into net interest margin; a steeper / higher long end widens spreads earned on loans funded with short-duration deposits.

**Specification search.** Initial OLS included `PPNR_lag2`, `Loans`, `Deposits`, `Nonperforming assets`, `Net charge-offs` as candidates. The final three-variable specification was selected after VIF pruning (some of those candidates exceeded VIF = 10 due to balance-sheet collinearity) and after a correlation-heatmap inspection.

**Diagnostics.** VIF on final regressors, residual-vs-fitted scatter, residual histogram, Q-Q plot for normality, and identification of the top-2 outlier quarters by absolute residual.

**Forecasting mechanics.** Recursive — `log(PPNR_{t-1})` is updated each quarter with the model's own previous prediction, `Total_Assets` is held flat at the last observed value (a simplifying assumption — the framework does not jointly forecast the balance sheet), and the 10Y yield comes from the Fed scenario path. PPNR is then exponentiated back to dollar levels for the capital engine.

#### 3b. Capital Projection Engine — Basel Expected-Loss Waterfall

The capital engine takes the four component forecasts (`retail_pd`, `wholesale_pd`, `wholesale_lgd`, `ppnr`) and applies a transparent regulatory accounting chain, run twice — once for baseline, once for severely adverse.

**Step 1 — Exposure at Default (EAD).** Industry-aggregate retail and wholesale loan balances from Fed H.8 are scaled to a single G-SIB via a market-share assumption:

```
EAD_retail    = $1,004B × MARKET_SHARE × retail_multiplier(scenario)
EAD_wholesale = $1,965B × MARKET_SHARE × wholesale_multiplier(scenario)
```

with `MARKET_SHARE = 0.12` (the G-SIB's share of total assets among the top U.S. banks). Scenario multipliers deflate exposures under stress to reflect deleveraging: `baseline = (1.00, 1.00)`, `adverse = (0.90, 0.80)`.

**Step 2 — Expected loss decomposition.** The textbook Basel formula, applied per quarter per portfolio:

```
retail_loss_t    = retail_pd_t × RETAIL_LGD × EAD_retail
wholesale_loss_t = wholesale_pd_t × wholesale_lgd_t × EAD_wholesale
total_loss_t     = retail_loss_t + wholesale_loss_t
```

with `RETAIL_LGD = 0.90` from the Fed supervisory prescription (the override discussed in §2b).

**Step 3 — 10-Q calibration anchor.** The model is anchored to the bank's most recently reported provision for credit losses to ensure the level of projected losses is realistic, not just the shape:

```
scaling_factor = target_loss / total_loss[0]
target_loss    = $2,849M   (JPM 2025Q2 10-Q provision for credit losses)
```

Every quarterly loss is multiplied by this scaling factor. Under severely adverse, an additional **front-loaded overlay** is applied to reproduce the supervisory loss-timing pattern in which the bulk of stress losses materialize in the first 6–8 quarters:

```
total_loss_t × 2.5   for t ∈ [0, 3]   (first 4 quarters - acute shock)
total_loss_t × 1.8   for t ∈ [4, 6]   (mid-cycle stress)
```

**Step 4 — PPNR stress overlay.** Under adverse, the regression-projected PPNR is compressed by a piecewise multiplier that decays toward 1.0 as the stress horizon advances:

```
PPNR_t × 0.45   for t ∈ [0, 3]
PPNR_t × 0.60   for t ∈ [4, 7]
PPNR_t × 0.75   for t ≥ 8
```

This mimics the Fed's revenue compression assumption (margin contraction, fee revenue decline, expense rigidity). These overlays are hand-tuned scalar multipliers rather than estimated parameters — a known scope limit.

**Step 5 — Net income waterfall:**

```
pre_tax_income_t = ppnr_t − total_loss_t
tax_t            = max(pre_tax_income_t, 0) × TAX_RATE
net_income_t     = (pre_tax_income_t − tax_t) × (1 − payout_ratio)
```

with `TAX_RATE = 3,297 / 18,284 ≈ 18%` (effective rate from the 10-Q: tax expense over pre-tax income) and `payout_ratio = 0.30` (30% of net income returned to shareholders as dividends, 70% retained). The `max(·, 0)` ensures no tax benefit is taken on a loss quarter.

**Step 6 — CET1 capital evolution** (recursive accumulation of retained earnings):

```
CET1_t = CET1_{t-1} + net_income_t
CET1_0 = $336.879B   (2025Q2 10-Q common stockholders' equity)
```

**Step 7 — Risk-Weighted Assets evolution** (compounded quarterly growth):

```
RWA_t = RWA_{t-1} × (1 + g)
RWA_0 = CET1_0 / 0.151   (back-solved from the reported 15.1% CET1 ratio)
g     = 0.000  (baseline)
      = 0.015  (adverse)
      = 0.040  (severely adverse, unused in shipped runs)
```

RWA growth under stress reflects rising risk weights as credit quality deteriorates (the same dollar exposure is weighted more heavily as borrowers migrate down the risk grid).

**Step 8 — CET1 ratio:**

```
CET1_ratio_t = CET1_t / RWA_t
```

The quarterly path is plotted against the **4.5% Basel III minimum** (the headline metric a regulator or capital planning committee would consume). The minimum CET1 ratio across the projection horizon is the project's principal stress-test output.

---

## Mathematical & Structural Mechanics

The framework is a one-directional pipeline that takes macro innovations and emits a capital trajectory. The flow is:

1. **Macro innovation.** A quarterly vector of supervisory macro factors (unemployment, GDP growth, spreads, yields, HPI, VIX) enters as an exogenous shock path.
2. **Component projection.** Each component model converts that vector into a portfolio-parameter path:
   - Retail PD path from the ECM, anchored on the long-run logit-DR equilibrium and pulled back by the error-correction term whenever the prior quarter deviated.
   - Wholesale LGD path from the OLS regression, with a recursive lag-1 feedback that produces persistence.
   - PPNR path from the log-linear autoregression, with assets and 10Y yield as the macro transmission channel into revenue.
3. **Expected loss aggregation.** Portfolio loss is the textbook Basel decomposition — probability of default times loss given default times exposure at default — applied independently to retail and wholesale and summed quarterly.
4. **Capital adequacy translation.** PPNR less losses produces pre-tax income; the bank's effective tax rate and dividend payout ratio scale this to retained earnings; retained earnings accumulate into CET1 capital while RWA grows at a scenario-specific compounding rate; the CET1 ratio is the quotient.
5. **Regulatory comparison.** The resulting quarterly CET1 path is compared against the 4.5% Basel III minimum, producing the headline metric a regulator or capital planning committee would consume.

The pipeline is deliberately **two-track**: the same code path runs once for the 2026 Supervisory Baseline and once for the Severely Adverse scenario, producing side-by-side projections that quantify the capital erosion attributable purely to the supervisory shock.

---

## Validation & Diagnostic Stack

| Check | Where | Purpose |
|---|---|---|
| Variance Inflation Factor (VIF) | Retail PD, Retail LGD, Wholesale LGD, PPNR | Detect multicollinearity (flag at 5, sever at 10) |
| Pairwise correlation cap | Wholesale LGD | Drop one of any pair with `|ρ| > 0.95`, with `prefer_keep` family priorities |
| Durbin-Watson & Ljung-Box (lags 4, 8) | Wholesale LGD | Residual autocorrelation diagnostics |
| Newey-West HAC standard errors | Wholesale LGD | Robust inference under residual autocorrelation |
| Economic sign check table | Wholesale LGD | Explicit `expected_sign` dictionary; coefficients flagged `OK` or `WRONG` |
| Two-step cointegration (Engle-Granger) | Retail PD | Implicit stationarity handling for the credit time series |
| LASSO 5-fold cross-validation | Retail LGD | Regularized feature selection on the lag pool |
| Residual diagnostics (resid-vs-fitted, histogram, Q-Q) | PPNR | Distributional check + outlier identification |

**Known scope limits.** No formal ADF/KPSS unit-root test, no walk-forward out-of-sample backtest, no explicit Vasicek single-factor asset-correlation calibration (exposure correlation is folded into the regression coefficients on stress indices), and no granular loan-level RWA modeling — RWA evolves at a scenario-specific scalar growth rate. The severely-adverse loss and PPNR overlays in the CET1 notebook are hand-tuned scalar multipliers rather than estimated parameters, calibrated to reproduce the front-loaded supervisory loss-timing pattern.

**Structural-break treatment.** Loan loss allowances and CET1 definitions in the historical panel are discontinuous across the **Basel III** and **CECL (ASU 2016-13)** transitions. Affected series were classified into pre / transition / post buckets and harmonized so coefficients are not driven by accounting-regime shifts.

---

## Technology Stack

| Layer | Libraries |
|---|---|
| Core data | `pandas`, `numpy` |
| Macro data ingestion | `pandas-datareader` (FRED), `datetime` |
| Econometrics | `statsmodels.api` (OLS, VIF, Newey-West HAC, Durbin-Watson, Ljung-Box), `statsmodels.formula.api` |
| Machine learning | `scikit-learn` (`LassoCV`, `StandardScaler`, `mean_squared_error`, `r2_score`) |
| Visualization | `matplotlib`, `seaborn`, `matplotlib.dates`, `matplotlib.ticker.PercentFormatter` |

# fim601-ccar

Minimal CCAR toolkit: scripts and notebooks to ingest FRED and Fed H.8 data, train component models (PD, LGD, PPNR), and run CET1 projections.

Quick start:
- python3 -m venv venv-ccar && source venv-ccar/bin/activate
- pip install -r requirements.txt
- python extract_macro_data.py
- jupyter nbconvert --to notebook --execute fed_data_DR.ipynb fed_data_CO.ipynb fed_data_AvgLoan.ipynb extract_retail_data.ipynb
- jupyter nbconvert --to notebook --execute retail_pd_model_v2.ipynb retail_lgd_model_v2.ipynb lgd_wholesale_model.ipynb PPNR_model.ipynb cet_forecast.ipynb

Outputs appear in ./output/ and ./outputs/. See filenames in repository.

License: educational/academic. Data from FRED and Federal Reserve.
