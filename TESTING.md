# Testing & Scientific Validation Suite

## 1. Overview of Automated Test Suite

The testing suite in [backend/tests/](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/tests/) validates every layer of the system—from CSV/Excel/API ingestion and missing-value tracking to PyMC SMC marginal likelihood estimation, BMA probability normalization, Law of Total Variance decomposition, outbreak risk classification, zero-leakage chronological cross-validation, and end-to-end FastAPI endpoints.

Run the complete test suite with:
```powershell
$env:PYTHONPATH="backend"; python -m pytest backend/tests/ -v
```

---

## 2. Test Modules & Verification Coverage

### 2.1 Phase 2: Data Pipeline, Missingness & EDA ([test_phase2_pipeline.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/tests/test_phase2_pipeline.py))
- `test_genuine_cdc_ilinet_dataset_ingestion`: Verifies that the genuine 2,502-row CDC ILINet surveillance dataset loads, validates, and persists into SQLAlchemy.
- `test_excel_ingestion_and_duplicate_and_missing_handling`: Verifies `.xlsx` spreadsheet ingestion, automatic deduplication of repeated dates, and explicit insertion of `is_missing=True` gap rows when weekly observations are omitted.
- `test_negative_counts_rejected`: Verifies that negative case counts are immediately rejected with a `ValueError`.
- `test_chronological_preprocessor_no_future_leakage`: Verifies that `ChronologicalPreprocessor` learns scaling parameters strictly from the training window and is completely unaffected by extreme spikes in the test window.
- `test_eda_module_on_genuine_data`: Verifies that `perform_exploratory_analysis` computes dispersion ratios ($\text{VMR} = \sigma^2/\mu$), seasonal week-of-year profiles, monthly trends, and multi-region comparisons on genuine CDC data.

### 2.2 Phase 3 & 4: Bayesian Models & BMA Mathematical Verification ([test_phase3_4_models_and_bma.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/tests/test_phase3_4_models_and_bma.py))
- `test_bma_probabilities_sum_to_one_and_nonnegative`: Verifies $\sum_{i=1}^K P(M_i \mid D) = 1.0$, $P(M_i \mid D) \ge 0$, and monotonic ordering with respect to marginal likelihood.
- `test_bma_numerical_stability_extreme_log_likelihoods`: Verifies that the Log-Sum-Exp implementation remains exact and free of underflow even when $\log P(D \mid M_i) \approx -150,000$.
- `test_convergence_failure_gating_excludes_nonconverged_model`: Verifies that a model flagged as non-converged (`converged=False`) receives $P(M_i \mid D) = 0.0$ even if its raw log marginal likelihood is high.
- `test_all_five_candidate_models_and_bma_on_genuine_data`: Trains all 5 PyMC models (`M1_POISSON`, `M2_NEGBIN`, `M3_AR_COUNT`, `M4_SEASONAL_FOURIER`, `M5_RENEWAL_GROWTH`) on genuine CDC NNDSS Pertussis surveillance data and verifies:
  - Finite SMC log marginal likelihoods across all models.
  - Strictly non-negative integer count predictions ($\tilde{y} \in \mathbb{N}_0$).
  - Proper quantile ordering ($Q_{0.025} \le Q_{0.10} \le \text{Median} \le Q_{0.90} \le Q_{0.975}$).
  - Exact **Law of Total Variance** identity: $\text{Total Variance} = \text{Within-Model Variance} + \text{Between-Model Variance}$.

### 2.3 Phase 5: Outbreak Detection, Proper Scoring Rules & Leakage Proof ([test_phase5_outbreak_and_eval.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/tests/test_phase5_outbreak_and_eval.py))
- `test_outbreak_risk_detector_exceedance_and_classifications`: Verifies analytical seasonal threshold computation, exceedance probability $P(Y_{T+h} > T_{\text{outbreak}} \mid D)$, and 4-tier risk classification (`Low`, `Moderate`, `High`, `Critical`).
- `test_crps_and_nlpd_proper_scoring_rules`: Verifies that CRPS and NLPD strictly reward sharp, centered predictive distributions over biased distributions.
- `test_expanding_window_backtest_no_leakage_on_genuine_data`: Executes a 2-fold expanding-window chronological backtest on genuine CDC NNDSS Campylobacteriosis data and **proves zero future leakage** by mutating held-out future observations to `999,999` and verifying that Fold 1 training log marginal likelihoods and BMA weights remain bit-for-bit identical.

### 2.4 Phase 6 & 8: Full-Stack FastAPI Integration ([test_phase6_api_integration.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/tests/test_phase6_api_integration.py))
- `test_health_and_datasets_endpoints`: Verifies `/health` and `/api/v1/datasets`.
- `test_diseases_locations_observations_and_eda_endpoints`: Verifies `/api/v1/diseases`, `/api/v1/locations`, `/api/v1/observations`, and `/api/v1/eda`.
- `test_model_analysis_forecasts_uncertainty_and_evaluation_endpoints`: Verifies `/api/v1/models/analysis`, `/api/v1/forecasts`, `/api/v1/uncertainty`, `/api/v1/evaluation`, and `/api/v1/overview`.
- `test_custom_dataset_upload_and_sync_training_workflow`: Tests the complete end-to-end user workflow: uploading a new CSV dataset via `POST /api/v1/datasets/upload`, triggering synchronous PyMC SMC training via `POST /api/v1/training/run?sync=true`, and retrieving the resulting BMA forecasts via `GET /api/v1/forecasts`.

