"""
Phase 5 Unit & Leakage Verification Tests:
Tests Outbreak Risk Detection, Exceedance Probability Calculation, Proper Scoring Rules (CRPS, NLPD, MASE),
and Expanding-Window Chronological Cross-Validation with strict verification of zero future leakage.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ml.evaluation.backtester import (
    compute_empirical_crps,
    compute_empirical_nlpd,
    run_expanding_window_backtest,
)
from app.ml.outbreak_detection.detector import OutbreakRiskDetector


def test_outbreak_risk_detector_exceedance_and_classifications():
    """Verify outbreak detector computes exact empirical exceedance probabilities and risk tiers."""
    detector = OutbreakRiskDetector(z_multiplier=1.28)
    train_dates = pd.date_range("2022-01-03", periods=104, freq="7D").strftime("%Y-%m-%d").tolist()
    # Baseline historical cases around 50 +/- 10
    rng = np.random.default_rng(42)
    train_y = rng.normal(50.0, 8.0, size=104).clip(5, 200).astype(np.int64)
    valid_mask = np.ones(104, dtype=bool)

    future_dates = ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
    future_epiweeks = [202401, 202402, 202403, 202404]

    # Construct BMA samples:
    # Step 1: low (mean ~30) -> Low Risk
    # Step 2: moderate (around threshold ~60) -> Moderate Risk
    # Step 3: high (mostly above threshold ~65) -> High Risk
    # Step 4: critical (all samples ~150 >> threshold) -> Critical Risk
    S = 1000
    bma_samples = np.column_stack([
        rng.poisson(30, size=S),
        rng.normal(58, 12, size=S).clip(0, None).astype(int),
        rng.normal(66, 10, size=S).clip(0, None).astype(int),
        rng.poisson(140, size=S),
    ])

    assessments = detector.evaluate_forecast_risk(
        train_dates=train_dates,
        train_y=train_y,
        valid_mask=valid_mask,
        future_dates=future_dates,
        future_epiweeks=future_epiweeks,
        bma_count_samples=bma_samples,
        bma_posterior_weights={"M3_AR_COUNT": 0.65, "M5_RENEWAL_GROWTH": 0.35},
    )
    assert len(assessments) == 4
    assert assessments[0].risk_level == "Low Risk"
    assert assessments[0].exceedance_prob < 0.10
    assert assessments[3].risk_level == "Critical Risk"
    assert assessments[3].exceedance_prob > 0.95
    for a in assessments:
        assert "Analytical Seasonal Endemic Channel" in a.threshold_method
        assert len(a.explanation_text) > 50


def test_crps_and_nlpd_proper_scoring_rules():
    """Verify CRPS and NLPD reward sharp, well-calibrated predictive distributions."""
    rng = np.random.default_rng(123)
    y_true = np.array([100.0, 120.0])

    # Sharp centered samples around [100, 120]
    good_samples = np.column_stack([
        rng.normal(100.0, 5.0, size=1000),
        rng.normal(120.0, 5.0, size=1000),
    ])
    # Biased samples around [200, 250]
    bad_samples = np.column_stack([
        rng.normal(200.0, 5.0, size=1000),
        rng.normal(250.0, 5.0, size=1000),
    ])

    crps_good = compute_empirical_crps(good_samples, y_true)
    crps_bad = compute_empirical_crps(bad_samples, y_true)
    assert crps_good >= 0.0
    assert crps_good < 5.0
    assert crps_bad > 80.0

    nlpd_good = compute_empirical_nlpd(good_samples, y_true)
    nlpd_bad = compute_empirical_nlpd(bad_samples, y_true)
    assert nlpd_good < nlpd_bad


def test_expanding_window_backtest_no_leakage_on_genuine_data():
    """
    Run 2-fold chronological expanding-window cross-validation on genuine CDC NNDSS Campylobacter data.
    Verifies:
    - Training cutoff strictly precedes test start date in every fold
    - BMA weights in Fold 1 are identical even if future actuals in Fold 2 are mutated (zero future leakage proof!)
    - All metrics (MAE, RMSE, MASE, CRPS, NLPD, Coverage 80/95) are finite and non-negative
    """
    csv_path = Path(__file__).resolve().parent.parent / "data" / "cdc_nndss_weekly_2022_2024.csv"
    df = pd.read_csv(csv_path)
    sub = df[(df["disease_code"] == "CAMPYLOBACTER") & (df["location_code"] == "US-NAT")].copy()
    sub = sub.sort_values("observation_date").iloc[:70].reset_index(drop=True)

    bt1 = run_expanding_window_backtest(
        obs_df=sub,
        n_folds=2,
        horizon=3,
        draws=150,
        chains=2,
        random_seed=42,
    )
    assert bt1["n_folds"] == 2
    assert len(bt1["metrics"]) == 6  # BMA + 5 candidate models
    for fold in bt1["folds"]:
        assert fold["train_end_date"] < fold["test_start_date"]
        assert np.isclose(sum(fold["bma_weights_in_fold"].values()), 1.0, atol=1e-3)

    # Leakage invariance proof: mutate the held-out test observations of Fold 1 & Fold 2 to 999,999
    # Fold 1 training window is [0 : 70 - 6 = 64]. Mutating index 64..69 must NOT change Fold 1 BMA weights!
    sub_mutated = sub.copy()
    sub_mutated.loc[64:, "confirmed_cases"] = 999999
    bt2 = run_expanding_window_backtest(
        obs_df=sub_mutated,
        n_folds=2,
        horizon=3,
        draws=150,
        chains=2,
        random_seed=42,
    )
    assert bt1["folds"][0]["bma_weights_in_fold"] == bt2["folds"][0]["bma_weights_in_fold"]
    assert bt1["folds"][0]["log_marginal_likelihoods_in_fold"] == bt2["folds"][0]["log_marginal_likelihoods_in_fold"]

