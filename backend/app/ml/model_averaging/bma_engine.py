"""
Genuine Bayesian Model Averaging (BMA) Engine.

Mathematical Foundation:
1. Posterior Model Probability:
   P(M_i | D) = [ P(D | M_i) * P(M_i) ] / sum_{j in V} [ P(D | M_j) * P(M_j) ]
   computed in log-space using log-sum-exp for numerical stability:
   log P(M_i | D) = (log P(D | M_i) + log P(M_i)) - logsumexp_{j in V}(log P(D | M_j) + log P(M_j))

2. Convergence Gating (Rule 10):
   Models that fail MCMC/SMC convergence (e.g., R-hat > threshold, ESS too low, or non-finite
   log marginal likelihood) are explicitly gated out of the active ensemble V (assigned P(M_i | D) = 0)
   with their failure reason documented.

3. Combined BMA Posterior Predictive Distribution:
   P(y_future | D) = sum_{i in V} P(y_future | D, M_i) * P(M_i | D)
   Preserves the discrete count mixture distribution by drawing joint trajectories from each
   converged candidate model's posterior predictive distribution proportional to P(M_i | D).

4. Exact Law of Total Variance Uncertainty Decomposition:
   Var(Y_{T+h} | D) = E_M[ Var(Y_{T+h} | D, M_i) ] + Var_M( E[Y_{T+h} | D, M_i] )
                    = Within-Model Variance (Parameter + Observation) + Between-Model Structural Variance.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp

from app.ml.bayesian_models.candidate_models import BayesianModelFitResult


@dataclass
class BMAEnsembleResult:
    model_codes: List[str]
    prior_probabilities: Dict[str, float]
    posterior_probabilities: Dict[str, float]
    log_marginal_likelihoods: Dict[str, float]
    converged_flags: Dict[str, bool]
    exclusion_reasons: Dict[str, Optional[str]]
    log_total_evidence: float
    # Combined discrete count samples: shape (n_bma_samples, horizon)
    bma_count_samples: np.ndarray
    horizon_mean: np.ndarray
    horizon_median: np.ndarray
    horizon_std: np.ndarray
    horizon_q025: np.ndarray
    horizon_q10: np.ndarray
    horizon_q25: np.ndarray
    horizon_q75: np.ndarray
    horizon_q90: np.ndarray
    horizon_q975: np.ndarray
    # Law of Total Variance 3-way decomposition across horizon steps:
    within_param_variance: np.ndarray
    within_obs_variance: np.ndarray
    within_total_variance: np.ndarray
    between_model_variance: np.ndarray
    total_predictive_variance: np.ndarray
    in_sample_bma_mean: np.ndarray
    in_sample_bma_q10: np.ndarray
    in_sample_bma_q90: np.ndarray


def compute_posterior_model_probabilities(
    log_marginal_likelihoods: Dict[str, float],
    converged_flags: Optional[Dict[str, bool]] = None,
    prior_probabilities: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Calculate normalized posterior model probabilities P(M_i | D) using log-sum-exp.

    Returns:
      - normalized_priors: P(M_i) over valid converged models
      - posterior_probs: P(M_i | D) summing strictly to 1.0
      - log_evidence: log P(D) = log sum_i P(D | M_i) P(M_i)
    """
    if not log_marginal_likelihoods:
        raise ValueError("log_marginal_likelihoods dictionary cannot be empty.")

    codes = list(log_marginal_likelihoods.keys())
    if converged_flags is None:
        converged_flags = {c: True for c in codes}

    # Identify valid models: must be marked converged AND have finite log marginal likelihood
    valid_codes = [
        c for c in codes
        if converged_flags.get(c, True) and np.isfinite(log_marginal_likelihoods[c])
    ]
    if not valid_codes:
        raise ValueError("No candidate models converged with finite marginal likelihood.")

    # Assign and normalize prior probabilities over valid candidate models
    raw_priors = {}
    for c in codes:
        if c not in valid_codes:
            raw_priors[c] = 0.0
        else:
            p_val = 1.0 if (prior_probabilities is None or c not in prior_probabilities) else float(prior_probabilities[c])
            if p_val < 0.0:
                raise ValueError(f"Prior model probability for {c} cannot be negative ({p_val}).")
            raw_priors[c] = p_val

    prior_sum = sum(raw_priors[c] for c in valid_codes)
    if prior_sum <= 0.0:
        raise ValueError("Sum of prior probabilities over valid models must be positive.")

    normalized_priors = {
        c: (raw_priors[c] / prior_sum if c in valid_codes else 0.0)
        for c in codes
    }

    # Log unnormalized posterior: log P(D | M_i) + log P(M_i)
    active_codes = [c for c in valid_codes if normalized_priors[c] > 0.0]
    log_unnorm = np.array(
        [log_marginal_likelihoods[c] + np.log(normalized_priors[c]) for c in active_codes],
        dtype=np.float64,
    )
    log_total_evidence = float(logsumexp(log_unnorm))
    log_post = log_unnorm - log_total_evidence
    post_arr = np.exp(log_post)
    # Exact renormalization to remove floating-point rounding drift (~1e-16)
    post_arr = post_arr / np.sum(post_arr)

    posterior_probs: Dict[str, float] = {c: 0.0 for c in codes}
    for idx, c in enumerate(active_codes):
        posterior_probs[c] = float(post_arr[idx])

    return normalized_priors, posterior_probs, log_total_evidence


def combine_posterior_predictive_distributions(
    model_results: List[BayesianModelFitResult],
    prior_probabilities: Optional[Dict[str, float]] = None,
    n_bma_samples: int = 2000,
    random_seed: int = 42,
) -> BMAEnsembleResult:
    """
    Combine candidate models into a genuine Bayesian Model Averaging ensemble.
    Constructs the full discrete posterior predictive mixture distribution and computes
    exact Law of Total Variance uncertainty decomposition.
    """
    if not model_results:
        raise ValueError("model_results list cannot be empty.")

    log_mls = {m.model_code: m.log_marginal_likelihood for m in model_results}
    conv_flags = {m.model_code: m.converged for m in model_results}
    excl_reasons = {m.model_code: m.exclusion_reason for m in model_results}

    norm_priors, post_probs, log_evidence = compute_posterior_model_probabilities(
        log_marginal_likelihoods=log_mls,
        converged_flags=conv_flags,
        prior_probabilities=prior_probabilities,
    )

    rng = np.random.default_rng(random_seed)
    horizon = model_results[0].future_count_samples.shape[1]
    n_in_sample = model_results[0].in_sample_mean.shape[0]

    # Allocate sample counts across models proportional to posterior probability P(M_i | D)
    # Using largest-remainder / multinomial mixture allocation so total draws == n_bma_samples
    codes = [m.model_code for m in model_results]
    weights = np.array([post_probs[c] for c in codes], dtype=np.float64)

    exact_quotas = weights * n_bma_samples
    int_quotas = np.floor(exact_quotas).astype(int)
    remainder = n_bma_samples - int(np.sum(int_quotas))
    if remainder > 0:
        frac_parts = exact_quotas - int_quotas
        top_indices = np.argsort(-frac_parts)[:remainder]
        for idx in top_indices:
            int_quotas[idx] += 1

    mixture_chunks = []
    for m_idx, m_res in enumerate(model_results):
        n_draw_i = int(int_quotas[m_idx])
        if n_draw_i <= 0:
            continue
        avail_samples = m_res.future_count_samples.shape[0]
        replace_flag = n_draw_i > avail_samples
        chosen_rows = rng.choice(avail_samples, size=n_draw_i, replace=replace_flag)
        mixture_chunks.append(m_res.future_count_samples[chosen_rows, :])

    bma_count_samples = np.vstack(mixture_chunks).astype(np.int64)
    # Shuffle rows so mixture draws are interleaved
    rng.shuffle(bma_count_samples, axis=0)

    # Analytical BMA mean & Law of Total Variance Decomposition across horizon
    # E[Y | D] = sum_i w_i E[Y | D, M_i]
    model_means = np.vstack([m.horizon_mean for m in model_results])  # shape (K, H)
    model_param_vars = np.vstack([m.param_variance for m in model_results])  # shape (K, H)
    model_obs_vars = np.vstack([m.obs_variance for m in model_results])  # shape (K, H)
    model_total_vars = np.vstack([m.total_variance for m in model_results])  # shape (K, H)

    w_col = weights[:, None]  # shape (K, 1)
    analytic_bma_mean = np.sum(w_col * model_means, axis=0)

    within_param_var = np.sum(w_col * model_param_vars, axis=0)
    within_obs_var = np.sum(w_col * model_obs_vars, axis=0)
    within_total_var = np.sum(w_col * model_total_vars, axis=0)

    # Between-model variance: sum_i w_i (E[Y | D, M_i] - E[Y | D])^2
    between_model_var = np.sum(w_col * ((model_means - analytic_bma_mean[None, :]) ** 2), axis=0)
    total_pred_var = within_total_var + between_model_var

    # In-sample BMA trajectory combination
    in_sample_means = np.vstack([m.in_sample_mean for m in model_results])
    in_sample_q10s = np.vstack([m.in_sample_q10 for m in model_results])
    in_sample_q90s = np.vstack([m.in_sample_q90 for m in model_results])

    in_sample_bma_mean = np.sum(w_col * in_sample_means, axis=0)
    in_sample_bma_q10 = np.sum(w_col * in_sample_q10s, axis=0)
    in_sample_bma_q90 = np.sum(w_col * in_sample_q90s, axis=0)

    return BMAEnsembleResult(
        model_codes=codes,
        prior_probabilities=norm_priors,
        posterior_probabilities=post_probs,
        log_marginal_likelihoods=log_mls,
        converged_flags=conv_flags,
        exclusion_reasons=excl_reasons,
        log_total_evidence=log_evidence,
        bma_count_samples=bma_count_samples,
        horizon_mean=analytic_bma_mean,
        horizon_median=np.median(bma_count_samples, axis=0),
        horizon_std=np.sqrt(np.maximum(total_pred_var, 0.0)),
        horizon_q025=np.percentile(bma_count_samples, 2.5, axis=0),
        horizon_q10=np.percentile(bma_count_samples, 10.0, axis=0),
        horizon_q25=np.percentile(bma_count_samples, 25.0, axis=0),
        horizon_q75=np.percentile(bma_count_samples, 75.0, axis=0),
        horizon_q90=np.percentile(bma_count_samples, 90.0, axis=0),
        horizon_q975=np.percentile(bma_count_samples, 97.5, axis=0),
        within_param_variance=within_param_var,
        within_obs_variance=within_obs_var,
        within_total_variance=within_total_var,
        between_model_variance=between_model_var,
        total_predictive_variance=total_pred_var,
        in_sample_bma_mean=in_sample_bma_mean,
        in_sample_bma_q10=in_sample_bma_q10,
        in_sample_bma_q90=in_sample_bma_q90,
    )

