# Disease Outbreak Prediction System Using Bayesian Model Averaging (BMA)

An end-to-end, research-grade epidemiological forecasting, uncertainty quantification, and outbreak risk classification platform built with **PyMC v6 Sequential Monte Carlo (SMC)**, **ArviZ**, **Statsmodels**, **FastAPI**, **SQLAlchemy**, and **React + TypeScript + Tailwind CSS + Recharts**.

---

## 1. Project Introduction & Problem Statement

Public health agencies rely on infectious disease surveillance forecasts to anticipate epidemic surges, allocate clinical resources, and evaluate outbreak risks. However, relying on any single predictive model introduces **structural model uncertainty**:
- A **Poisson regression model** enforces equidispersion ($\text{Var}(Y_t) = \mathbb{E}[Y_t]$) and severely underestimates predictive interval widths during overdispersed outbreaks.
- A **purely seasonal harmonic model** captures multi-year endemic cycles well but lags behind sudden unseasonal transmission surges.
- An **autoregressive or renewal model** reacts quickly to recent week-over-week momentum but can over-extrapolate transient reporting spikes.

This project addresses single-model vulnerability through **genuine Bayesian Model Averaging (BMA)**—weighting competing Bayesian models by their posterior model probabilities $P(M_i \mid D)$ computed from Sequential Monte Carlo (SMC) marginal likelihoods $\log P(D \mid M_i)$ and combining their full discrete posterior predictive distributions.

---

## 2. Objectives

1. **Ingest & Validate Genuine Epidemiological Surveillance Data**: Provide pre-bundled and live-API access to official U.S. CDC ILINet (`ILI`) and CDC NNDSS (`DENGUE`, `PERTUSSIS`, `SALMONELLOSIS`, `CAMPYLOBACTER`, `MALARIA`) surveillance time series alongside configurable CSV and Excel (`.xlsx`) ingestion.
2. **Enforce Strict Temporal Causality & Data Integrity**: Never silently ignore missing observations, never perform random train-test splitting on chronological series, and never leak future observations into preprocessing scalers, autoregressive lags, or BMA weights.
3. **Train 5 Statistically Distinct Bayesian Candidate Models**: Estimate posterior parameter distributions and marginal likelihoods via PyMC SMC (`pm.sample_smc`) with ArviZ convergence gating ($\hat{R}$, Bulk/Tail ESS).
4. **Synthesize Genuine BMA Posterior Mixtures & 3-Way Variance Decomposition**: Combine full discrete count distributions $P(y_{\text{future}} \mid D) = \sum_i P(y_{\text{future}} \mid D, M_i) P(M_i \mid D)$ and decompose total predictive variance via the Law of Total Variance into **Parameter Uncertainty**, **Aleatoric Observation Noise**, and **Structural Between-Model Uncertainty**.
5. **Quantify Outbreak Risk**: Calculate posterior exceedance probabilities $P(Y_{T+h} > T_{\text{outbreak}} \mid D)$ relative to an analytical Farrington-style seasonal endemic channel.
6. **Evaluate Objectively via Expanding-Window Chronological Cross-Validation**: Compare BMA against every individual candidate model using count-safe proper scoring rules (CRPS, NLPD, MASE, RMSE, MAE, and 80%/95% interval coverage).

---

## 3. System Architecture & Technology Stack

See [ARCHITECTURE.md](file:///c:/Users/lokes/Downloads/ML%20presentation/ARCHITECTURE.md) for the complete component diagram and database entity specifications.

- **Programming Language**: Python 3.11+ (tested on Python 3.14) & TypeScript 5.6
- **Bayesian & Statistical Computing**: `PyMC 6.4`, `ArviZ 1.3`, `Statsmodels 0.15`, `SciPy`, `NumPy`, `Pandas`, `Scikit-learn`
- **Backend API & ORM**: `FastAPI`, `Pydantic v2`, `SQLAlchemy 2.1`, `Uvicorn`
- **Database**: `SQLite` (pre-seeded for zero-config local execution) & `PostgreSQL 16` (via Docker Compose)
- **Frontend Dashboard**: `React 18`, `TypeScript`, `Vite`, `Tailwind CSS`, `Recharts`, `Lucide Icons`

---

## 4. Dataset Sources & Preprocessing Methodology

See [DATA_SOURCES.md](file:///c:/Users/lokes/Downloads/ML%20presentation/DATA_SOURCES.md) and [METHODOLOGY.md](file:///c:/Users/lokes/Downloads/ML%20presentation/METHODOLOGY.md) for full details.

- **Primary Dataset (`cdc_ilinet_weekly_2017_2024.csv`)**: 2,502 weekly observations of Influenza-Like Illness (`num_ili`) and patient visit denominators (`num_patients`) across `US-NAT` and 5 HHS regions (`US-HHS1`, `US-HHS2`, `US-HHS4`, `US-HHS6`, `US-HHS9`) from the U.S. CDC Outpatient Influenza-like Illness Surveillance Network via the CMU Delphi Epidata API.
- **Secondary Multi-Disease Dataset (`cdc_nndss_weekly_2022_2024.csv`)**: 7,020 weekly records covering `DENGUE`, `PERTUSSIS`, `SALMONELLOSIS`, `CAMPYLOBACTER`, and `MALARIA` across 9 U.S. jurisdictions (`US-NAT`, `US-PR`, `US-FL`, `US-CA`, `US-NY`, etc.) from the CDC National Notifiable Diseases Surveillance System (NNDSS) Socrata API.

---

## 5. Mathematical Foundation of BMA & Candidate Models

See [MATHEMATICAL_FOUNDATION.md](file:///c:/Users/lokes/Downloads/ML%20presentation/MATHEMATICAL_FOUNDATION.md) for full derivations.

### Candidate Models ([candidate_models.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/bayesian_models/candidate_models.py))
- **`M1_POISSON`**: Bayesian Poisson GLM with linear trend and fundamental annual harmonics.
- **`M2_NEGBIN`**: Bayesian Negative Binomial Regression estimating overdispersion $\alpha > 0$ ($\text{Var}(Y_t) = \mu_t + \mu_t^2/\alpha$).
- **`M3_AR_COUNT`**: Bayesian Autoregressive Count Model (`Log-AR(2)`) with recursive posterior trajectory rollout.
- **`M4_SEASONAL_FOURIER`**: Bayesian Multi-Harmonic Seasonal Model ($K=3$ Fourier harmonics with shrinkage priors).
- **`M5_RENEWAL_GROWTH`**: Mechanistic-inspired Bayesian Renewal Model ($\mathbb{E}[Y_t] = I_{t-1} \cdot \mathcal{R}_t$) with serial-interval infectious load and density-dependent damping.

### Bayesian Model Averaging ([bma_engine.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/model_averaging/bma_engine.py))
Posterior model weights are computed from SMC marginal likelihoods $\ell_i = \log P(D \mid M_i)$:
$$P(M_i \mid D) = \frac{P(D \mid M_i) P(M_i)}{\sum_{j \in \mathcal{V}} P(D \mid M_j) P(M_j)}, \qquad P(\mathbf{y}_{\text{future}} \mid D) = \sum_{i \in \mathcal{V}} P(\mathbf{y}_{\text{future}} \mid D, M_i) P(M_i \mid D)$$

---

## 6. Outbreak Detection & Model Evaluation Summary

See [MODEL_EVALUATION.md](file:///c:/Users/lokes/Downloads/ML%20presentation/MODEL_EVALUATION.md) for complete chronological cross-validation tables across diseases.

- On **U.S. National Influenza-Like Illness (`ILI`, `US-NAT`)**, the BMA ensemble achieved the lowest out-of-sample probabilistic error (**$\text{CRPS} = 12,133.08$**, **$\text{RMSE} = 26,902.62$**, **$\text{MASE} = 3.293$**, **95% Coverage = $100.0\%$**), outperforming every individual candidate model.
- On **U.S. National Pertussis (`PERTUSSIS`, `US-NAT`)**, BMA allocated $59.19\%$ posterior weight to `M5_RENEWAL_GROWTH` and $40.81\%$ to `M3_AR_COUNT`, achieving $\text{CRPS} = 72.25$ ($87.5\%$ empirical coverage at the $80\%$ level and $100.0\%$ at the $95\%$ level), outperforming 4 of the 5 individual models while honest backtest reporting shows `M5_RENEWAL_GROWTH` achieved slightly lower point error ($\text{CRPS} = 63.96$) during that specific post-cutoff surge.
- On **U.S. National Campylobacteriosis (`CAMPYLOBACTER`, `US-NAT`)**, BMA achieved the best out-of-sample **Negative Log Predictive Density ($\text{NLPD} = 6.60$)** across all models.

---

## 7. Installation & Execution Instructions

### 7.1 Local Execution (Without Docker)

1. **Install Python Dependencies**:
   ```powershell
   pip install -r backend/requirements.txt
   ```
2. **Verify or Seed Database (Pre-seeded SQLite database is already included in `backend/data/bma_outbreak_system.db`)**:
   ```powershell
   $env:PYTHONPATH="backend"
   python backend/scripts/seed_initial_runs.py
   ```
3. **Start the FastAPI Backend Server (Port 8000)**:
   ```powershell
   $env:PYTHONPATH="backend"
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
4. **Start the React + TypeScript Dashboard (Port 5173)**:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

### 7.2 Docker & Docker Compose Execution
```bash
docker compose up --build
```

---

## 8. REST API Documentation

Interactive OpenAPI / Swagger documentation is served automatically at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Service health check |
| `/api/v1/datasets` | `GET` | List registered datasets, SHA-256 hashes, and provenance |
| `/api/v1/datasets/upload` | `POST` | Upload and normalize custom CSV or Excel (`.xlsx`) datasets |
| `/api/v1/diseases` | `GET` | List available diseases and their associated geographic locations |
| `/api/v1/locations` | `GET` | List available locations (optionally filtered by `disease_code`) |
| `/api/v1/observations` | `GET` | Retrieve chronological observations and explicit missingness flags |
| `/api/v1/eda` | `GET` | Run comprehensive EDA, seasonal decomposition, VMR dispersion, and anomaly detection |
| `/api/v1/training/run` | `POST` | Trigger a new PyMC SMC + BMA + Outbreak Risk + Chronological Backtest run |
| `/api/v1/training/runs/{id}/status` | `GET` | Poll progress percentage and status message of a training run |
| `/api/v1/models/analysis` | `GET` | Retrieve candidate model SMC marginal likelihoods, ArviZ diagnostics, and BMA weights |
| `/api/v1/forecasts` | `GET` | Retrieve BMA and individual model forecasts, 80%/95% intervals, and outbreak risk tiers |
| `/api/v1/uncertainty` | `GET` | Retrieve 3-way Law of Total Variance uncertainty decomposition and model entropy |
| `/api/v1/evaluation` | `GET` | Retrieve expanding-window chronological backtest metrics and fold-by-fold plots |
| `/api/v1/overview` | `GET` | Unified summary payload for the Overview intelligence page |

---

## 9. Interactive Dashboard Pages (Screenshots & Capabilities)

1. **Page 1 — Overview**: Displays selected disease/region, data freshness, latest reported cases, historical incidence vs. BMA forecast chart, BMA weight bar chart, and horizon risk table.
2. **Page 2 — Data Explorer**: Supports custom CSV/Excel file uploads, historical date filtering, weekly/monthly trends, 4w/12w moving averages, ISO week seasonal profiles, additive decomposition, geographic comparisons, and outlier audits.
3. **Page 3 — Model Analysis**: Displays all 5 candidate Bayesian models, SMC log marginal likelihoods $\log P(D \mid M_i) \pm \text{SE}$, WAIC/LOO, ArviZ $\hat{R}$ and ESS diagnostics, posterior parameter tables, and in-sample fits.
4. **Page 4 — Outbreak Forecast**: Allows users to configure forecast horizon, SMC draws/chains, threshold multiplier, and random seed to trigger live PyMC training, toggle individual candidate model trajectories against the BMA forecast, and inspect exceedance probabilities $P(Y_{T+h} > T_{\text{outbreak}} \mid D)$.
5. **Page 5 — Uncertainty Analysis**: Visualizes the 80% and 95% predictive interval fan chart, the 3-way Law of Total Variance decomposition (Parameter vs. Aleatoric Observation vs. Structural Model Variance), discrete posterior predictive histograms, and prior-to-posterior model probability concentration.
6. **Page 6 — Model Evaluation**: Presents expanding-window chronological backtest results (CRPS, NLPD, MAE, RMSE, MASE, 80%/95% coverage) and an interactive fold-by-fold out-of-sample forecast inspector.
7. **Page 7 — About the Model**: Documents the mathematical equations, conceptual distinctions, and epistemic scope of the system.

---

## 10. Limitations & Future Enhancements

### Limitations
- **Provisional Surveillance Reporting Delays**: Public health surveillance records are subject to administrative backfill; forecasts are conditioned on provisional reported counts at time $T$.
- **Unobserved Exogenous Interventions**: Purely statistical and renewal time-series models cannot anticipate sudden policy shifts, novel viral strain emergence, or changes in clinical diagnostic definitions prior to their reflection in surveillance counts.
- **Computational Cost of SMC**: Sequential Monte Carlo tempering across 5 models + multi-fold backtesting requires several seconds to minutes depending on particle count $N_{\text{particles}}$.

### Future Enhancements
- **Hierarchical Spatial-Temporal BMA**: Jointly modeling cross-region spatial transmission matrices across adjacent states/HHS regions.
- **Reporting Delay Nowcasting Integration**: Coupling a Bayesian reporting-triangle nowcast model with the BMA forecasting ensemble.
- **Covariate Integration**: Incorporating meteorological series (absolute humidity, temperature, precipitation) for vector-borne diseases such as Dengue.

