"""
Phase 3 & Phase 4 Unit and Mathematical Verification Tests:
Tests all 5 Bayesian candidate models (Poisson, NegBin, AR-Count, Seasonal Fourier, Renewal Growth),
SMC marginal likelihood estimation, convergence gating, posterior model probability normalization,
log-sum-exp numerical stability under extreme log-evidence values, discrete posterior predictive
mixture combination, and exact Law of Total Variance decomposition.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ml.bayesian_models.candidate_models import CandidateModelSuite
from app.ml.model_averaging.bma_engine import (
    combine_posterior_predictive_distributions,
    compute_posterior_model_probabilities,
)
from app.ml.preprocessing.preprocessor import ChronologicalPreprocessor


def test_bma_probabilities_sum_to_one_and_nonnegative():
    """Mathematical Test 1, 2, 3: Posterior model probabilities are nonnegative and sum to 1."""
    log_mls = {
        "M1_POISSON": -2840.5,
        "M2_NEGBIN": -1120.3,
        "M3_AR_COUNT": -1118.9,
        "M4_SEASONAL_FOURIER": -1121.7,
        "M5_RENEWAL_GROWTH": -1119.4,
    }
    priors, posts, log_ev = compute_posterior_model_probabilities(log_mls)

    assert np.isclose(sum(priors.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(posts.values()), 1.0, atol=1e-12)
    for c, w in posts.items():
        assert 0.0 <= w <= 1.0
    # M3_AR_COUNT has highest log marginal likelihood (-1118.9), so it must have highest posterior weight
    assert posts["M3_AR_COUNT"] > posts["M5_RENEWAL_GROWTH"] > posts["M2_NEGBIN"] > posts["M1_POISSON"]


def test_bma_numerical_stability_extreme_log_likelihoods():
    """Verify log-sum-exp safeguard handles extremely small marginal likelihoods (-10^5) without underflow/NaN."""
    extreme_log_mls = {
        "M1": -150000.0,
        "M2": -149998.0,
        "M3": -149999.0,
    }
    _, posts, log_ev = compute_posterior_model_probabilities(extreme_log_mls)
    assert np.isfinite(log_ev)
    assert np.isclose(sum(posts.values()), 1.0, atol=1e-12)
    # Analytical check: exp(-149998) / (exp(-150000) + exp(-149999) + exp(-149998))
    expected_m2 = np.exp(2.0) / (np.exp(0.0) + np.exp(1.0) + np.exp(2.0))
    assert np.isclose(posts["M2"], expected_m2, atol=1e-9)


def test_convergence_failure_gating_excludes_nonconverged_model():
    """Rule 10 & Section 7: Models failing convergence are excluded from BMA weight allocation."""
    log_mls = {
        "M1_POISSON": -500.0,
        "M2_NEGBIN": -450.0,  # Higher marginal likelihood, but failed convergence!
        "M3_AR_COUNT": -460.0,
    }
    conv = {
        "M1_POISSON": True,
        "M2_NEGBIN": False,
        "M3_AR_COUNT": True,
    }
    priors, posts, _ = compute_posterior_model_probabilities(log_mls, converged_flags=conv)
    assert posts["M2_NEGBIN"] == 0.0
    assert priors["M2_NEGBIN"] == 0.0
    assert np.isclose(posts["M1_POISSON"] + posts["M3_AR_COUNT"], 1.0, atol=1e-12)


def test_all_five_candidate_models_and_bma_on_genuine_data():
    """
    End-to-end test of all 5 candidate Bayesian models + BMA mixture on genuine CDC NNDSS Pertussis data.
    Verifies:
    - All 5 models train via PyMC SMC and return finite marginal likelihoods
    - Count predictions are strictly non-negative integers
    - 80% interval is nested inside 95% interval
    - Law of Total Variance holds: Total Variance == Within-Model Variance + Between-Model Variance
    - Overdispersed count models receive higher evidence than equidispersed Poisson on overdispersed data
    """
    csv_path = Path(__file__).resolve().parent.parent / "data" / "cdc_nndss_weekly_2022_2024.csv"
    df = pd.read_csv(csv_path)
    sub = df[(df["disease_code"] == "PERTUSSIS") & (df["location_code"] == "US-NAT")].copy()
    sub = sub.sort_values("observation_date").iloc[:80].reset_index(drop=True)

    prep = ChronologicalPreprocessor(seasonality_period=52.18)
    train_feats, prep_state = prep.fit_transform_train(sub)
    fut_covs = prep.build_future_deterministic_covariates(horizon=4)

    suite = CandidateModelSuite(draws=250, chains=2, random_seed=42)
    results = suite.fit_all_models(
        train_features=train_feats,
        prep_state=prep_state,
        future_covariates=fut_covs,
    )
    assert len(results) == 5

    for r in results:
        assert np.isfinite(r.log_marginal_likelihood)
        assert r.future_count_samples.shape == (500, 4)
        # Mathematical Test 6: Count predictions remain nonnegative integers
        assert np.all(r.future_count_samples >= 0)
        assert np.issubdtype(r.future_count_samples.dtype, np.integer)
        # Mathematical Test 5: Predictive intervals are ordered properly
        assert np.all(r.horizon_q025 <= r.horizon_q10)
        assert np.all(r.horizon_q10 <= r.horizon_median)
        assert np.all(r.horizon_median <= r.horizon_q90)
        assert np.all(r.horizon_q90 <= r.horizon_q975)

    bma = combine_posterior_predictive_distributions(results, n_bma_samples=1000, random_seed=42)
    assert np.isclose(sum(bma.posterior_probabilities.values()), 1.0, atol=1e-12)
    assert bma.bma_count_samples.shape == (1000, 4)
    assert np.all(bma.bma_count_samples >= 0)

    # Verify Law of Total Variance identity
    assert np.allclose(
        bma.total_predictive_variance,
        bma.within_total_variance + bma.between_model_variance,
        rtol=1e-10,
    )
    # Verify BMA favors overdispersed Negative Binomial models over Poisson on overdispersed data
    assert bma.posterior_probabilities["M1_POISSON"] < 0.05

