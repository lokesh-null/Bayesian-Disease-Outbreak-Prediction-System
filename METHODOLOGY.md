# Scientific Methodology: Epidemiological Preprocessing, Bayesian Inference & Outbreak Detection

## 1. Data Ingestion & Missing-Observation Integrity

Epidemiological surveillance streams from public health agencies frequently exhibit reporting irregularities, column naming variations, duplicate provisional updates, and missing weekly reports. Our ingestion layer ([ingestor.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/data_ingestion/ingestor.py)) implements a multi-stage validation pipeline:

1. **Configurable Schema Mapping**: Automatically maps source columns across CDC, WHO, and custom CSV/Excel spreadsheets into a canonical internal schema (`observation_date`, `epiweek`, `disease_code`, `location_code`, `confirmed_cases`, `deaths`, `exposure_denominator`, `population`).
2. **Domain Validation**: Rejects negative count values and invalid dates immediately.
3. **Deduplication**: Resolves duplicate provisional reports for the same `(disease_code, location_code, observation_date)` by retaining the latest revision while recording the exact `duplicate_count`.
4. **Explicit Chronological Gap Detection**: For weekly surveillance series, the pipeline inspects consecutive observation timestamps. Whenever a week is omitted in the raw file, an explicit record with `confirmed_cases = NULL` and `is_missing = True` is inserted so missing weeks are never silently ignored.

---

## 2. Leakage-Safe Chronological Preprocessing

All temporal scaling and feature engineering operations ([preprocessor.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/preprocessing/preprocessor.py)) obey strict temporal causality:

- **No Random Splitting**: Train/test splits are strictly chronological (`train_df` covers $t \le T_{\text{cutoff}}$, `test_df` covers $t > T_{\text{cutoff}}$).
- **Training-Only Scalers**: The time index mean $\mu_t$, standard deviation $\sigma_t$, baseline mean $\bar{y}_{\text{train}}$, and log-incidence centering constant $\overline{\log(y_{\text{train}} + 1)}$ are computed exclusively on the training window $[1, T_{\text{cutoff}}]$.
- **Causal Lag Handling**: If a training window contains missing observations (`is_missing = True`), the likelihood is evaluated only at valid observed time steps (`valid_mask`), while autoregressive lag covariates use strictly causal forward-fill (`ffill`) from past observations—never backward-filling from future observations.
- **Recursive Trajectory Rollout for Dynamic Models**: For $M_3$ (`M3_AR_COUNT`) and $M_5$ (`M5_RENEWAL_GROWTH`), multi-step forecasts ($h = 1, \dots, H$) never reference future actuals. At horizon step $T+h$, each posterior sample $s \in \{1, \dots, S\}$ computes its lagged predictors from its own previously sampled count $\tilde{y}_{T+h-1}^{(s)}$.

---

## 3. Convergence Monitoring & Gating

Bayesian inference via PyMC's Sequential Monte Carlo sampler (`pm.sample_smc`) is monitored using ArviZ (`az.summary`) across multiple independent chains ([candidate_models.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/bayesian_models/candidate_models.py)):

- **Gelman-Rubin Statistic ($\hat{R}$)**: Measures between-chain vs. within-chain variance. Any model with $\max \hat{R} > 1.10$ is flagged as non-converged.
- **Effective Sample Size ($\text{ESS}_{\text{bulk}}, \text{ESS}_{\text{tail}}$)**: Models with $\min \text{ESS}_{\text{bulk}} < 40$ are flagged as non-converged.
- **Marginal Likelihood Finity**: All chains must yield finite $\log \hat{P}(D \mid M_i)$.
- **Ensemble Gating**: If a candidate model fails any convergence check, it is automatically excluded from the BMA ensemble ($P(M_i \mid D) = 0$) and its exclusion reason is persisted in `candidate_model_results.exclusion_reason` and displayed in the UI.

---

## 4. Outbreak Detection & Analytical Risk Classification

The outbreak detection module ([detector.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/app/ml/outbreak_detection/detector.py)) implements a Farrington-inspired **Analytical Seasonal Endemic Channel**:

1. **Seasonal Baseline Estimation**: For target ISO week $w \in \{1, \dots, 52\}$, the detector extracts all historical training observations falling within a circular $\pm 3$-week seasonal window ($\min(|w_t - w|, 52 - |w_t - w|) \le 3$) and computes the historical seasonal mean $\mu_{\text{hist}}(w)$, standard deviation $\sigma_{\text{hist}}(w)$, and 75th percentile floor $Q_{0.75}(w)$.
2. **Analytical Outbreak Threshold**:
   $$T_{\text{outbreak}}(w) = \max\left( \mu_{\text{hist}}(w) + z_\alpha \cdot \sigma_{\text{hist}}(w), \; Q_{0.75}(w), \; 1.0 \right)$$
   where $z_\alpha = 1.28$ by default (configurable in the UI between $0.5$ and $3.5$). Because official statutory epidemic thresholds vary by jurisdiction, all thresholds are explicitly labeled as analytical surveillance thresholds.
3. **Posterior Exceedance Probability**:
   Using the $S_{\text{BMA}} = 2000$ discrete count draws from the combined BMA posterior predictive distribution at horizon step $T+h$:
   $$P(Y_{T+h} > T_{\text{outbreak}} \mid D) = \frac{1}{S_{\text{BMA}}} \sum_{s=1}^{S_{\text{BMA}}} \mathbb{I}\left( \tilde{y}_{T+h}^{(s)} > T_{\text{outbreak}}(w_{T+h}) \right)$$
4. **Risk Classification Tiers**:
   - **Low Risk**: $P_{\text{exceed}} < 0.25$
   - **Moderate Risk**: $0.25 \le P_{\text{exceed}} < 0.55$ (or $P_{\text{exceed}} \ge 0.18$ with expected growth rate $> +35\%$)
   - **High Risk**: $0.55 \le P_{\text{exceed}} < 0.80$
   - **Critical Risk**: $P_{\text{exceed}} \ge 0.80$

