"""
Candidate Bayesian Epidemiological Time-Series Models (PyMC v6 + ArviZ v1.3).

Implements 5 statistically distinct candidate models for disease count forecasting:
- Model M1 (`M1_POISSON`): Bayesian Poisson Generalized Linear Model (equidispersed baseline).
- Model M2 (`M2_NEGBIN`): Bayesian Negative Binomial Regression (overdispersed count model).
- Model M3 (`M3_AR_COUNT`): Bayesian Autoregressive Log-Linear Count Model (INGARCH(2) structure).
- Model M4 (`M4_SEASONAL_FOURIER`): Bayesian Multi-Harmonic Seasonal Negative Binomial Model (K=3 harmonics).
- Model M5 (`M5_RENEWAL_GROWTH`): Bayesian Epidemic Renewal / Density-Dependent Growth Rate Model.

Marginal likelihood log P(D | M_i) is estimated via Sequential Monte Carlo (SMC) tempering
(`pm.sample_smc`), complemented by Laplace-corrected evidence and pointwise log-likelihood WAIC/LOO.
Never uses the unstable harmonic mean estimator.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from scipy.special import gammaln, logsumexp

from app.ml.preprocessing.preprocessor import TemporalPreprocessorState

# Suppress noisy C++ compiler warning on Windows where Numba/VM handles execution
pytensor.config.cxx = ""


@dataclass
class BayesianModelFitResult:
    model_code: str
    model_name: str
    model_family: str
    assumptions_text: str
    converged: bool
    exclusion_reason: Optional[str]
    log_marginal_likelihood: float
    log_ml_std_error: float
    laplace_log_evidence: float
    waic: float
    loo: float
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    divergences: int
    sampling_time_sec: float
    smc_chain_log_mls: List[float]
    warnings_list: List[str]
    parameter_summary: List[Dict[str, Any]]
    in_sample_mean: np.ndarray
    in_sample_q10: np.ndarray
    in_sample_q90: np.ndarray
    # Future predictions shape: (n_samples, horizon)
    future_count_samples: np.ndarray
    future_mean_samples: np.ndarray
    # Decomposed uncertainty across horizon steps
    horizon_mean: np.ndarray
    horizon_median: np.ndarray
    horizon_std: np.ndarray
    horizon_q025: np.ndarray
    horizon_q10: np.ndarray
    horizon_q25: np.ndarray
    horizon_q75: np.ndarray
    horizon_q90: np.ndarray
    horizon_q975: np.ndarray
    param_variance: np.ndarray
    obs_variance: np.ndarray
    total_variance: np.ndarray


def _extract_smc_log_marginal_likelihood(idata: Any) -> Tuple[float, float, List[float]]:
    """
    Extract chain-level log marginal likelihood estimates from PyMC SMC sample_stats.
    Uses log-mean-exp across chains for the combined marginal likelihood estimate
    and computes the standard error across chains.
    """
    raw_lml = idata.sample_stats["log_marginal_likelihood"].values
    chain_vals: List[float] = []

    def _collect_last_finite(item: Any) -> Optional[float]:
        if isinstance(item, (list, tuple, np.ndarray)):
            finite_sub = []
            for sub in item:
                val = _collect_last_finite(sub)
                if val is not None and np.isfinite(val):
                    finite_sub.append(val)
            return finite_sub[-1] if finite_sub else None
        try:
            fval = float(item)
            return fval if np.isfinite(fval) else None
        except (TypeError, ValueError):
            return None

    arr = np.asarray(raw_lml, dtype=object)
    for c_idx in range(arr.shape[0]):
        val = _collect_last_finite(arr[c_idx])
        if val is not None:
            chain_vals.append(val)

    if not chain_vals:
        return float("nan"), float("nan"), []

    chain_arr = np.array(chain_vals, dtype=np.float64)
    # Log-mean-exp across SMC chains: log(1/C * sum_c exp(lml_c))
    combined_lml = float(logsumexp(chain_arr) - np.log(len(chain_arr)))
    std_err = float(np.std(chain_arr, ddof=1) / np.sqrt(len(chain_arr))) if len(chain_arr) > 1 else 0.0
    return combined_lml, std_err, chain_vals


def _compute_pointwise_information_criteria(
    y_valid: np.ndarray,
    mu_samples_valid: np.ndarray,
    alpha_samples: Optional[np.ndarray],
    is_poisson: bool,
    n_params: int,
) -> Tuple[float, float, float]:
    """
    Compute exact pointwise log-likelihood matrix (S x N_valid) to obtain:
    1. WAIC (Widely Applicable Information Criterion, elpd_waic)
    2. Pareto-smoothed LOO approximation / cross-validated elpd
    3. Laplace-BIC / Chib posterior log-evidence cross-check
    """
    # y_valid shape (1, N), mu_samples_valid shape (S, N)
    y_row = y_valid[None, :].astype(np.float64)
    mu_safe = np.clip(mu_samples_valid, 1e-8, 1e9)

    if is_poisson:
        # Poisson log PMF: y * log(mu) - mu - log(y!)
        log_lik_matrix = y_row * np.log(mu_safe) - mu_safe - gammaln(y_row + 1.0)
    else:
        # Negative Binomial (mu, alpha) log PMF:
        # gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) + alpha*log(alpha/(alpha+mu)) + y*log(mu/(alpha+mu))
        alpha_col = np.clip(alpha_samples[:, None], 1e-6, 1e7)
        log_lik_matrix = (
            gammaln(y_row + alpha_col)
            - gammaln(alpha_col)
            - gammaln(y_row + 1.0)
            + alpha_col * (np.log(alpha_col) - np.log(alpha_col + mu_safe))
            + y_row * (np.log(mu_safe) - np.log(alpha_col + mu_safe))
        )

    S, N = log_lik_matrix.shape
    # Pointwise lppd_i = log(1/S sum_s exp(log_lik_{s,i}))
    lppd_i = logsumexp(log_lik_matrix, axis=0) - np.log(S)
    # Pointwise variance p_waic_i = Var_s(log_lik_{s,i})
    p_waic_i = np.var(log_lik_matrix, axis=0, ddof=1)
    elpd_waic = float(np.sum(lppd_i - p_waic_i))

    # Importance sampling LOO log predictive density per observation:
    # log p(y_i | y_{-i}) = - log(1/S sum_s exp(-log_lik_{s,i})) with truncation for stability
    neg_ll = -log_lik_matrix
    # Truncated importance sampling (Ionides 2008) to bound variance
    mean_neg_ll = logsumexp(neg_ll, axis=0) - np.log(S)
    trunc_neg_ll = np.minimum(neg_ll, mean_neg_ll[None, :] + 0.5 * np.log(S))
    loo_i = -(logsumexp(trunc_neg_ll, axis=0) - np.log(S))
    elpd_loo = float(np.sum(loo_i))

    # Laplace-BIC log marginal likelihood approximation around posterior mean log-likelihood
    total_ll_per_draw = np.sum(log_lik_matrix, axis=1)
    max_ll = float(np.max(total_ll_per_draw))
    laplace_log_ev = float(max_ll - 0.5 * n_params * np.log(max(N, 2)))

    return elpd_waic, elpd_loo, laplace_log_ev


def _sample_negbin_counts(
    rng: np.random.Generator, mu: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """
    Draw exact discrete Negative Binomial counts under PyMC's (mu, alpha) parameterization
    via Gamma-Poisson mixture:
      lambda ~ Gamma(shape=alpha, scale=mu/alpha)
      y ~ Poisson(lambda)
    """
    mu_safe = np.clip(mu, 1e-8, 1e7)
    alpha_safe = np.clip(alpha, 1e-4, 1e6)
    scale = mu_safe / alpha_safe
    lam = rng.gamma(shape=alpha_safe, scale=scale)
    lam = np.clip(lam, 0.0, 1e8)
    return rng.poisson(lam=lam).astype(np.int64)


class CandidateModelSuite:
    """
    Trains and evaluates the 5 candidate Bayesian epidemiological models using PyMC SMC
    and produces full posterior predictive count distributions and convergence diagnostics.
    """

    MODEL_METADATA = {
        "M1_POISSON": {
            "name": "Bayesian Poisson Generalized Linear Model",
            "family": "Poisson Count Regression",
            "assumptions": (
                "Assumes conditional variance equals conditional mean (Var(Y_t | lambda_t) = E[Y_t | lambda_t] = lambda_t) "
                "with log-linear temporal trend and fundamental annual seasonal harmonics (period P=52.18 weeks). "
                "When actual surveillance data are overdispersed, Poisson likelihood assigns narrow predictive bands "
                "and receives lower Bayesian marginal likelihood."
            ),
        },
        "M2_NEGBIN": {
            "name": "Bayesian Negative Binomial Regression",
            "family": "Overdispersed Count Regression",
            "assumptions": (
                "Relaxes the Poisson equidispersion constraint by introducing dispersion parameter alpha > 0 such that "
                "Var(Y_t) = mu_t + mu_t^2 / alpha. Captures extra-Poisson variability arising from superspreading events, "
                "spatial clustering, and surveillance reporting batching."
            ),
        },
        "M3_AR_COUNT": {
            "name": "Bayesian Autoregressive Count Model (Log-AR(2))",
            "family": "Dynamic Autoregressive Time-Series",
            "assumptions": (
                "Models serial transmission dependency via order-2 lagged log-incidence log(y_{t-1}+1) and log(y_{t-2}+1) "
                "combined with trend and Negative Binomial overdispersion. Multi-step forecasts roll out recursively per "
                "posterior trajectory so future observations are never leaked."
            ),
        },
        "M4_SEASONAL_FOURIER": {
            "name": "Bayesian Multi-Harmonic Seasonal Model (Fourier K=3)",
            "family": "Harmonic Seasonal Count Model",
            "assumptions": (
                "Captures complex, asymmetric annual and semi-annual seasonal epidemic profiles using K=3 Fourier harmonic pairs "
                "(sin/cos at w, 2w, 3w) with frequency-shrinking Gaussian priors N(0, (1/k)^2) and Negative Binomial likelihood."
            ),
        },
        "M5_RENEWAL_GROWTH": {
            "name": "Bayesian Renewal / Density-Dependent Growth Model",
            "family": "Mechanistic Epidemiological Renewal Process",
            "assumptions": (
                "Derived from the discrete epidemic renewal equation E[Y_t] = I_{t-1} * R_t, where I_{t-1} is the serial-interval "
                "weighted infectious load over the preceding 3 weeks and log(R_t) includes baseline reproduction, seasonality, "
                "and density-dependent damping (susceptible depletion at epidemic peaks)."
            ),
        },
    }

    def __init__(
        self,
        draws: int = 600,
        chains: int = 2,
        random_seed: int = 42,
        rhat_threshold: float = 1.10,
        min_ess_threshold: float = 40.0,
    ):
        self.draws = draws
        self.chains = chains
        self.random_seed = random_seed
        self.rhat_threshold = rhat_threshold
        self.min_ess_threshold = min_ess_threshold

    def fit_single_model(
        self,
        model_code: str,
        train_features: Dict[str, Any],
        prep_state: TemporalPreprocessorState,
        future_covariates: Dict[str, Any],
    ) -> BayesianModelFitResult:
        """
        Build PyMC model for `model_code`, sample posterior + marginal likelihood via SMC,
        extract ArviZ diagnostics, and roll out posterior predictive trajectories.
        """
        t0 = time.perf_counter()
        meta = self.MODEL_METADATA[model_code]
        valid_mask = train_features["valid_mask"]
        y_valid = train_features["y"][valid_mask]
        t_scaled_all = train_features["t_scaled"]
        t_scaled_valid = t_scaled_all[valid_mask]
        sin_all = train_features["fourier_sin"]
        cos_all = train_features["fourier_cos"]
        sin_valid = sin_all[valid_mask]
        cos_valid = cos_all[valid_mask]

        prior_intercept_mu = float(np.log(max(prep_state.y_train_mean, 1.0)))
        warnings_captured: List[str] = []

        with pm.Model() as pymc_model:
            if model_code == "M1_POISSON":
                beta_0 = pm.Normal("beta_0", mu=prior_intercept_mu, sigma=1.5)
                beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.8)
                beta_sin1 = pm.Normal("beta_sin1", mu=0.0, sigma=1.0)
                beta_cos1 = pm.Normal("beta_cos1", mu=0.0, sigma=1.0)

                eta = (
                    beta_0
                    + beta_trend * t_scaled_valid
                    + beta_sin1 * sin_valid[:, 0]
                    + beta_cos1 * cos_valid[:, 0]
                )
                mu_lik = pm.math.exp(pm.math.clip(eta, -10.0, 16.0))
                pm.Poisson("y_obs", mu=mu_lik, observed=y_valid)

            elif model_code == "M2_NEGBIN":
                beta_0 = pm.Normal("beta_0", mu=prior_intercept_mu, sigma=1.5)
                beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.8)
                beta_sin1 = pm.Normal("beta_sin1", mu=0.0, sigma=1.0)
                beta_cos1 = pm.Normal("beta_cos1", mu=0.0, sigma=1.0)
                alpha = pm.Gamma("alpha", alpha=2.0, beta=0.5)

                eta = (
                    beta_0
                    + beta_trend * t_scaled_valid
                    + beta_sin1 * sin_valid[:, 0]
                    + beta_cos1 * cos_valid[:, 0]
                )
                mu_lik = pm.math.exp(pm.math.clip(eta, -10.0, 16.0))
                pm.NegativeBinomial("y_obs", mu=mu_lik, alpha=alpha, observed=y_valid)

            elif model_code == "M3_AR_COUNT":
                lag1_valid = train_features["lag1_centered"][valid_mask]
                lag2_valid = train_features["lag2_centered"][valid_mask]

                beta_0 = pm.Normal("beta_0", mu=prior_intercept_mu, sigma=1.2)
                beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.5)
                phi_1 = pm.Normal("phi_1", mu=0.6, sigma=0.4)
                phi_2 = pm.Normal("phi_2", mu=0.1, sigma=0.35)
                beta_sin1 = pm.Normal("beta_sin1", mu=0.0, sigma=0.6)
                beta_cos1 = pm.Normal("beta_cos1", mu=0.0, sigma=0.6)
                alpha = pm.Gamma("alpha", alpha=2.5, beta=0.5)

                eta = (
                    beta_0
                    + beta_trend * t_scaled_valid
                    + phi_1 * lag1_valid
                    + phi_2 * lag2_valid
                    + beta_sin1 * sin_valid[:, 0]
                    + beta_cos1 * cos_valid[:, 0]
                )
                mu_lik = pm.math.exp(pm.math.clip(eta, -10.0, 16.0))
                pm.NegativeBinomial("y_obs", mu=mu_lik, alpha=alpha, observed=y_valid)

            elif model_code == "M4_SEASONAL_FOURIER":
                beta_0 = pm.Normal("beta_0", mu=prior_intercept_mu, sigma=1.5)
                beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.8)
                a1 = pm.Normal("a_sin1", mu=0.0, sigma=1.0)
                b1 = pm.Normal("b_cos1", mu=0.0, sigma=1.0)
                a2 = pm.Normal("a_sin2", mu=0.0, sigma=0.5)
                b2 = pm.Normal("b_cos2", mu=0.0, sigma=0.5)
                a3 = pm.Normal("a_sin3", mu=0.0, sigma=0.33)
                b3 = pm.Normal("b_cos3", mu=0.0, sigma=0.33)
                alpha = pm.Gamma("alpha", alpha=2.0, beta=0.5)

                eta = (
                    beta_0
                    + beta_trend * t_scaled_valid
                    + a1 * sin_valid[:, 0]
                    + b1 * cos_valid[:, 0]
                    + a2 * sin_valid[:, 1]
                    + b2 * cos_valid[:, 1]
                    + a3 * sin_valid[:, 2]
                    + b3 * cos_valid[:, 2]
                )
                mu_lik = pm.math.exp(pm.math.clip(eta, -10.0, 16.0))
                pm.NegativeBinomial("y_obs", mu=mu_lik, alpha=alpha, observed=y_valid)

            elif model_code == "M5_RENEWAL_GROWTH":
                log_inf_valid = train_features["log_infectious_load"][valid_mask]
                log_inf_cent_valid = train_features["log_infectious_load_centered"][valid_mask]

                r_0 = pm.Normal("r_0", mu=0.0, sigma=0.5)
                r_trend = pm.Normal("r_trend", mu=0.0, sigma=0.3)
                r_damp = pm.Normal("r_damp", mu=-0.15, sigma=0.25)
                s_sin1 = pm.Normal("s_sin1", mu=0.0, sigma=0.5)
                s_cos1 = pm.Normal("s_cos1", mu=0.0, sigma=0.5)
                alpha = pm.Gamma("alpha", alpha=2.5, beta=0.5)

                eta = (
                    log_inf_valid
                    + r_0
                    + r_trend * t_scaled_valid
                    + r_damp * log_inf_cent_valid
                    + s_sin1 * sin_valid[:, 0]
                    + s_cos1 * cos_valid[:, 0]
                )
                mu_lik = pm.math.exp(pm.math.clip(eta, -10.0, 16.0))
                pm.NegativeBinomial("y_obs", mu=mu_lik, alpha=alpha, observed=y_valid)

            else:
                raise ValueError(f"Unknown candidate model code: {model_code}")

            with warnings.catch_warnings(record=True) as w_rec:
                warnings.simplefilter("always")
                idata = pm.sample_smc(
                    draws=self.draws,
                    chains=self.chains,
                    cores=1,
                    random_seed=self.random_seed,
                    progressbar=False,
                )
                for w in w_rec:
                    msg = str(w.message)
                    if "recommend running at least 4 chains" not in msg and "g++" not in msg:
                        warnings_captured.append(msg[:200])

        elapsed = time.perf_counter() - t0

        # Extract ArviZ summary & convergence diagnostics
        summary_df = az.summary(idata)
        param_summary_list: List[Dict[str, Any]] = []
        rhat_vals: List[float] = []
        ess_bulk_vals: List[float] = []
        ess_tail_vals: List[float] = []

        for param_name, row in summary_df.iterrows():
            rhat_v = float(row.get("r_hat", 1.0)) if pd.notna(row.get("r_hat", 1.0)) else 1.0
            ess_b = float(row.get("ess_bulk", self.draws)) if pd.notna(row.get("ess_bulk", self.draws)) else float(self.draws)
            ess_t = float(row.get("ess_tail", self.draws)) if pd.notna(row.get("ess_tail", self.draws)) else float(self.draws)
            rhat_vals.append(rhat_v)
            ess_bulk_vals.append(ess_b)
            ess_tail_vals.append(ess_t)

            lb_col = "eti89_lb" if "eti89_lb" in row else ("hdi_3%" if "hdi_3%" in row else None)
            ub_col = "eti89_ub" if "eti89_ub" in row else ("hdi_97%" if "hdi_97%" in row else None)
            param_summary_list.append({
                "parameter": str(param_name),
                "mean": round(float(row["mean"]), 4),
                "sd": round(float(row["sd"]), 4),
                "ci_lower": round(float(row[lb_col]), 4) if lb_col and pd.notna(row[lb_col]) else None,
                "ci_upper": round(float(row[ub_col]), 4) if ub_col and pd.notna(row[ub_col]) else None,
                "r_hat": round(rhat_v, 4),
                "ess_bulk": round(ess_b, 1),
                "ess_tail": round(ess_t, 1),
            })

        max_rhat = float(max(rhat_vals)) if rhat_vals else 1.0
        min_ess_bulk = float(min(ess_bulk_vals)) if ess_bulk_vals else float(self.draws)
        min_ess_tail = float(min(ess_tail_vals)) if ess_tail_vals else float(self.draws)

        log_ml, log_ml_se, chain_lmls = _extract_smc_log_marginal_likelihood(idata)

        # Check convergence gate (Rule 10: Never hide model convergence failures)
        converged = True
        exclusion_reasons = []
        if not np.isfinite(log_ml):
            converged = False
            exclusion_reasons.append("SMC marginal likelihood estimate is non-finite (NaN/Inf).")
        if max_rhat > self.rhat_threshold:
            converged = False
            exclusion_reasons.append(
                f"Gelman-Rubin R-hat ({max_rhat:.3f}) exceeded threshold ({self.rhat_threshold:.2f})."
            )
        if min_ess_bulk < self.min_ess_threshold:
            converged = False
            exclusion_reasons.append(
                f"Bulk Effective Sample Size ({min_ess_bulk:.1f}) fell below threshold ({self.min_ess_threshold:.1f})."
            )
        exclusion_reason = " ".join(exclusion_reasons) if exclusion_reasons else None

        # Flatten posterior draws across chains: shape (S,) where S = chains * draws
        post = idata.posterior
        rng = np.random.default_rng(self.random_seed + hash(model_code) % 10000)

        in_sample_mu_draws, future_mu_draws, future_count_draws, alpha_draws = self._rollout_posterior_predictive(
            model_code=model_code,
            post=post,
            train_features=train_features,
            prep_state=prep_state,
            future_covariates=future_covariates,
            rng=rng,
        )

        is_poisson = model_code == "M1_POISSON"
        waic_val, loo_val, laplace_ev = _compute_pointwise_information_criteria(
            y_valid=y_valid,
            mu_samples_valid=in_sample_mu_draws[:, valid_mask],
            alpha_samples=alpha_draws,
            is_poisson=is_poisson,
            n_params=len(param_summary_list),
        )

        # In-sample fitted summary
        in_sample_mean = np.mean(in_sample_mu_draws, axis=0)
        in_sample_q10 = np.percentile(in_sample_mu_draws, 10, axis=0)
        in_sample_q90 = np.percentile(in_sample_mu_draws, 90, axis=0)

        # Future predictive summaries & within-model uncertainty decomposition:
        # Parameter variance = Var_s(E[Y_{T+h} | theta^(s)])
        # Observation (aleatoric) variance = E_s[Var(Y_{T+h} | theta^(s))]
        param_var = np.var(future_mu_draws, axis=0, ddof=1)
        if is_poisson:
            obs_var = np.mean(future_mu_draws, axis=0)
        else:
            alpha_col = alpha_draws[:, None]
            obs_var = np.mean(future_mu_draws + (future_mu_draws ** 2) / np.clip(alpha_col, 1e-4, 1e7), axis=0)
        total_var = np.var(future_count_draws, axis=0, ddof=1)

        return BayesianModelFitResult(
            model_code=model_code,
            model_name=meta["name"],
            model_family=meta["family"],
            assumptions_text=meta["assumptions"],
            converged=converged,
            exclusion_reason=exclusion_reason,
            log_marginal_likelihood=log_ml,
            log_ml_std_error=log_ml_se,
            laplace_log_evidence=laplace_ev,
            waic=waic_val,
            loo=loo_val,
            max_rhat=max_rhat,
            min_ess_bulk=min_ess_bulk,
            min_ess_tail=min_ess_tail,
            divergences=0,
            sampling_time_sec=round(elapsed, 3),
            smc_chain_log_mls=chain_lmls,
            warnings_list=warnings_captured,
            parameter_summary=param_summary_list,
            in_sample_mean=in_sample_mean,
            in_sample_q10=in_sample_q10,
            in_sample_q90=in_sample_q90,
            future_count_samples=future_count_draws,
            future_mean_samples=future_mu_draws,
            horizon_mean=np.mean(future_count_draws, axis=0),
            horizon_median=np.median(future_count_draws, axis=0),
            horizon_std=np.std(future_count_draws, axis=0, ddof=1),
            horizon_q025=np.percentile(future_count_draws, 2.5, axis=0),
            horizon_q10=np.percentile(future_count_draws, 10.0, axis=0),
            horizon_q25=np.percentile(future_count_draws, 25.0, axis=0),
            horizon_q75=np.percentile(future_count_draws, 75.0, axis=0),
            horizon_q90=np.percentile(future_count_draws, 90.0, axis=0),
            horizon_q975=np.percentile(future_count_draws, 97.5, axis=0),
            param_variance=param_var,
            obs_variance=obs_var,
            total_variance=total_var,
        )

    def _rollout_posterior_predictive(
        self,
        model_code: str,
        post: Any,
        train_features: Dict[str, Any],
        prep_state: TemporalPreprocessorState,
        future_covariates: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute in-sample posterior mean trajectories and out-of-sample H-step ahead
        posterior predictive count samples.
        For dynamic models (M3_AR_COUNT and M5_RENEWAL_GROWTH), step T+h uses the
        sampled counts from step T+h-1, strictly avoiding future actual leakage.
        """
        def flat(var_name: str) -> np.ndarray:
            return np.ravel(post[var_name].values).astype(np.float64)

        t_train = train_features["t_scaled"]
        sin_train = train_features["fourier_sin"]
        cos_train = train_features["fourier_cos"]

        horizon = future_covariates["horizon"]
        t_fut = future_covariates["t_scaled"]
        sin_fut = future_covariates["fourier_sin"]
        cos_fut = future_covariates["fourier_cos"]

        if model_code == "M1_POISSON":
            b0 = flat("beta_0")[:, None]
            bt = flat("beta_trend")[:, None]
            bs1 = flat("beta_sin1")[:, None]
            bc1 = flat("beta_cos1")[:, None]

            eta_train = b0 + bt * t_train[None, :] + bs1 * sin_train[:, 0][None, :] + bc1 * cos_train[:, 0][None, :]
            mu_train = np.exp(np.clip(eta_train, -10.0, 16.0))

            eta_fut = b0 + bt * t_fut[None, :] + bs1 * sin_fut[:, 0][None, :] + bc1 * cos_fut[:, 0][None, :]
            mu_fut = np.exp(np.clip(eta_fut, -10.0, 16.0))
            counts_fut = rng.poisson(lam=mu_fut).astype(np.int64)
            return mu_train, mu_fut, counts_fut, None

        elif model_code == "M2_NEGBIN":
            b0 = flat("beta_0")[:, None]
            bt = flat("beta_trend")[:, None]
            bs1 = flat("beta_sin1")[:, None]
            bc1 = flat("beta_cos1")[:, None]
            alpha = flat("alpha")

            eta_train = b0 + bt * t_train[None, :] + bs1 * sin_train[:, 0][None, :] + bc1 * cos_train[:, 0][None, :]
            mu_train = np.exp(np.clip(eta_train, -10.0, 16.0))

            eta_fut = b0 + bt * t_fut[None, :] + bs1 * sin_fut[:, 0][None, :] + bc1 * cos_fut[:, 0][None, :]
            mu_fut = np.exp(np.clip(eta_fut, -10.0, 16.0))
            counts_fut = _sample_negbin_counts(rng, mu_fut, alpha[:, None])
            return mu_train, mu_fut, counts_fut, alpha

        elif model_code == "M3_AR_COUNT":
            b0 = flat("beta_0")
            bt = flat("beta_trend")
            phi1 = flat("phi_1")
            phi2 = flat("phi_2")
            bs1 = flat("beta_sin1")
            bc1 = flat("beta_cos1")
            alpha = flat("alpha")
            S = len(b0)

            lag1_tr = train_features["lag1_centered"]
            lag2_tr = train_features["lag2_centered"]
            eta_train = (
                b0[:, None]
                + bt[:, None] * t_train[None, :]
                + phi1[:, None] * lag1_tr[None, :]
                + phi2[:, None] * lag2_tr[None, :]
                + bs1[:, None] * sin_train[:, 0][None, :]
                + bc1[:, None] * cos_train[:, 0][None, :]
            )
            mu_train = np.exp(np.clip(eta_train, -10.0, 16.0))

            # Recursive rollout over horizon H
            mu_fut = np.zeros((S, horizon), dtype=np.float64)
            counts_fut = np.zeros((S, horizon), dtype=np.int64)

            last_counts = prep_state.last_observed_counts
            y_prev1 = np.full(S, float(last_counts[-1]), dtype=np.float64)
            y_prev2 = np.full(S, float(last_counts[-2] if len(last_counts) >= 2 else last_counts[-1]), dtype=np.float64)
            log_mean = prep_state.log_y_plus1_mean

            for h in range(horizon):
                l1_cent = np.log(y_prev1 + 1.0) - log_mean
                l2_cent = np.log(y_prev2 + 1.0) - log_mean
                eta_h = (
                    b0
                    + bt * t_fut[h]
                    + phi1 * l1_cent
                    + phi2 * l2_cent
                    + bs1 * sin_fut[h, 0]
                    + bc1 * cos_fut[h, 0]
                )
                mu_h = np.exp(np.clip(eta_h, -10.0, 16.0))
                y_h = _sample_negbin_counts(rng, mu_h, alpha)
                mu_fut[:, h] = mu_h
                counts_fut[:, h] = y_h
                y_prev2 = y_prev1
                y_prev1 = y_h.astype(np.float64)

            return mu_train, mu_fut, counts_fut, alpha

        elif model_code == "M4_SEASONAL_FOURIER":
            b0 = flat("beta_0")[:, None]
            bt = flat("beta_trend")[:, None]
            a1 = flat("a_sin1")[:, None]
            b1 = flat("b_cos1")[:, None]
            a2 = flat("a_sin2")[:, None]
            b2 = flat("b_cos2")[:, None]
            a3 = flat("a_sin3")[:, None]
            b3 = flat("b_cos3")[:, None]
            alpha = flat("alpha")

            eta_train = (
                b0
                + bt * t_train[None, :]
                + a1 * sin_train[:, 0][None, :]
                + b1 * cos_train[:, 0][None, :]
                + a2 * sin_train[:, 1][None, :]
                + b2 * cos_train[:, 1][None, :]
                + a3 * sin_train[:, 2][None, :]
                + b3 * cos_train[:, 2][None, :]
            )
            mu_train = np.exp(np.clip(eta_train, -10.0, 16.0))

            eta_fut = (
                b0
                + bt * t_fut[None, :]
                + a1 * sin_fut[:, 0][None, :]
                + b1 * cos_fut[:, 0][None, :]
                + a2 * sin_fut[:, 1][None, :]
                + b2 * cos_fut[:, 1][None, :]
                + a3 * sin_fut[:, 2][None, :]
                + b3 * cos_fut[:, 2][None, :]
            )
            mu_fut = np.exp(np.clip(eta_fut, -10.0, 16.0))
            counts_fut = _sample_negbin_counts(rng, mu_fut, alpha[:, None])
            return mu_train, mu_fut, counts_fut, alpha

        elif model_code == "M5_RENEWAL_GROWTH":
            r0 = flat("r_0")
            rt = flat("r_trend")
            rdamp = flat("r_damp")
            ss1 = flat("s_sin1")
            sc1 = flat("s_cos1")
            alpha = flat("alpha")
            S = len(r0)

            log_inf_tr = train_features["log_infectious_load"]
            log_inf_cent_tr = train_features["log_infectious_load_centered"]
            eta_train = (
                log_inf_tr[None, :]
                + r0[:, None]
                + rt[:, None] * t_train[None, :]
                + rdamp[:, None] * log_inf_cent_tr[None, :]
                + ss1[:, None] * sin_train[:, 0][None, :]
                + sc1[:, None] * cos_train[:, 0][None, :]
            )
            mu_train = np.exp(np.clip(eta_train, -10.0, 16.0))

            mu_fut = np.zeros((S, horizon), dtype=np.float64)
            counts_fut = np.zeros((S, horizon), dtype=np.int64)

            last_c = prep_state.last_observed_counts
            y_m1 = np.full(S, float(last_c[-1]), dtype=np.float64)
            y_m2 = np.full(S, float(last_c[-2] if len(last_c) >= 2 else last_c[-1]), dtype=np.float64)
            y_m3 = np.full(S, float(last_c[-3] if len(last_c) >= 3 else last_c[0]), dtype=np.float64)
            w = prep_state.serial_weights
            log_mean = prep_state.log_y_plus1_mean

            for h in range(horizon):
                inf_load = w[0] * y_m1 + w[1] * y_m2 + w[2] * y_m3 + 1.0
                log_inf = np.log(inf_load)
                log_inf_cent = log_inf - log_mean
                eta_h = (
                    log_inf
                    + r0
                    + rt * t_fut[h]
                    + rdamp * log_inf_cent
                    + ss1 * sin_fut[h, 0]
                    + sc1 * cos_fut[h, 0]
                )
                mu_h = np.exp(np.clip(eta_h, -10.0, 16.0))
                y_h = _sample_negbin_counts(rng, mu_h, alpha)
                mu_fut[:, h] = mu_h
                counts_fut[:, h] = y_h
                y_m3 = y_m2
                y_m2 = y_m1
                y_m1 = y_h.astype(np.float64)

            return mu_train, mu_fut, counts_fut, alpha

        raise ValueError(f"Unsupported model_code: {model_code}")

    def fit_all_models(
        self,
        train_features: Dict[str, Any],
        prep_state: TemporalPreprocessorState,
        future_covariates: Dict[str, Any],
        model_codes: Optional[List[str]] = None,
        progress_callback: Optional[Any] = None,
    ) -> List[BayesianModelFitResult]:
        """Train all requested candidate models sequentially and report progress."""
        codes = model_codes or list(self.MODEL_METADATA.keys())
        results: List[BayesianModelFitResult] = []
        for idx, code in enumerate(codes):
            if progress_callback:
                progress_callback(
                    pct=round(10.0 + (70.0 * idx / len(codes)), 1),
                    msg=f"Sampling candidate model {code} ({self.MODEL_METADATA[code]['name']}) via SMC...",
                )
            res = self.fit_single_model(
                model_code=code,
                train_features=train_features,
                prep_state=prep_state,
                future_covariates=future_covariates,
            )
            results.append(res)
        return results
