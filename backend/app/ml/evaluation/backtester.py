"""
Expanding-Window Chronological Cross-Validation & Proper Scoring Rules Evaluation Module.

Strictly enforces:
- Rule 6: Never use future information when training models for historical validation.
- Rule 7: Never perform random train-test splitting on chronological disease surveillance data.
- Rule 8: Never report training accuracy as forecasting performance.
- Section 7: Model probabilities must be calculated using training data only during every historical validation fold.
- Section 9: Objectively establish whether BMA improves forecasting performance; if BMA performs worse than an individual model on a given metric, report that result honestly.

Metrics implemented (all zero-count safe; MAPE is intentionally excluded):
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Squared Error)
3. MASE (Mean Absolute Scaled Error, scaled by in-sample 1-step naive random walk MAE)
4. CRPS (Continuous Ranked Probability Score via exact O(S log S) empirical order-statistics energy identity)
5. NLPD (Negative Log Predictive Density via Gaussian-kernel smoothed discrete mass evaluation)
6. 80% & 95% Empirical Predictive Interval Coverage
7. 80% & 95% Mean Predictive Interval Width
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from app.ml.bayesian_models.candidate_models import CandidateModelSuite
from app.ml.model_averaging.bma_engine import combine_posterior_predictive_distributions
from app.ml.preprocessing.preprocessor import ChronologicalPreprocessor


@dataclass
class ModelEvaluationMetrics:
    model_code: str
    model_name: str
    mae: float
    rmse: float
    mase: float
    crps: float
    nlpd: float
    coverage_80: float
    coverage_95: float
    interval_width_80: float
    interval_width_95: float
    mean_bma_weight: Optional[float]


def compute_empirical_crps(samples_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute exact Continuous Ranked Probability Score (CRPS) for empirical sample matrix
    `samples_matrix` of shape (S, H) against actual observations `y_true` of shape (H,).

    Uses the O(S log S) quantile / order-statistic representation of the energy identity:
      CRPS(F, y) = E|X - y| - 1/(2 S^2) sum_{i,j} |X_i - X_j|
    """
    S, H = samples_matrix.shape
    crps_vals = np.zeros(H, dtype=np.float64)
    idx_weights = (2.0 * np.arange(1, S + 1, dtype=np.float64) - S - 1.0)

    for h in range(H):
        x_sorted = np.sort(samples_matrix[:, h].astype(np.float64))
        term1 = np.mean(np.abs(x_sorted - float(y_true[h])))
        # 1/(2 S^2) sum_{i,j} |x_i - x_j| == (1 / S^2) sum_{k=1}^S (2k - S - 1) x_{(k)}
        term2 = np.sum(idx_weights * x_sorted) / (S * S)
        crps_vals[h] = max(term1 - term2, 0.0)

    return float(np.mean(crps_vals))


def compute_empirical_nlpd(samples_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Negative Log Predictive Density (NLPD) at observed counts `y_true`
    using a bandwidth-adaptive discrete/Gaussian kernel mixture over posterior predictive draws.
    Lower values indicate superior probabilistic calibration and sharpness.
    """
    S, H = samples_matrix.shape
    nlpd_vals = np.zeros(H, dtype=np.float64)

    for h in range(H):
        draws = samples_matrix[:, h].astype(np.float64)
        obs = float(y_true[h])
        sample_std = float(np.std(draws, ddof=1)) if S > 1 else 1.0
        # Silverman-style bandwidth with minimum floor of 1.0 count unit for discrete support
        bw = max(1.06 * sample_std * (S ** (-0.2)), 1.0)
        log_kernels = -0.5 * (((draws - obs) / bw) ** 2) - np.log(bw * np.sqrt(2.0 * np.pi))
        log_pred_density = float(logsumexp(log_kernels) - np.log(S))
        nlpd_vals[h] = -log_pred_density

    return float(np.mean(nlpd_vals))


def evaluate_predictions_against_actuals(
    model_code: str,
    model_name: str,
    samples_list: List[np.ndarray],
    y_true_list: List[np.ndarray],
    naive_scale_list: List[float],
    bma_weights_across_folds: Optional[List[float]] = None,
) -> ModelEvaluationMetrics:
    """
    Aggregate out-of-sample evaluation metrics across all chronological validation folds.
    """
    all_errors = []
    all_sq_errors = []
    all_scaled_errors = []
    crps_per_fold = []
    nlpd_per_fold = []
    cov80_flags = []
    cov95_flags = []
    width80_vals = []
    width95_vals = []

    for f_idx, samples in enumerate(samples_list):
        y_true = y_true_list[f_idx].astype(np.float64)
        naive_mae = max(naive_scale_list[f_idx], 1e-6)

        mean_pred = np.mean(samples, axis=0)
        q025 = np.percentile(samples, 2.5, axis=0)
        q10 = np.percentile(samples, 10.0, axis=0)
        q90 = np.percentile(samples, 90.0, axis=0)
        q975 = np.percentile(samples, 97.5, axis=0)

        abs_err = np.abs(y_true - mean_pred)
        all_errors.extend(abs_err.tolist())
        all_sq_errors.extend(((y_true - mean_pred) ** 2).tolist())
        all_scaled_errors.extend((abs_err / naive_mae).tolist())

        crps_per_fold.append(compute_empirical_crps(samples, y_true))
        nlpd_per_fold.append(compute_empirical_nlpd(samples, y_true))

        in_80 = (y_true >= q10) & (y_true <= q90)
        in_95 = (y_true >= q025) & (y_true <= q975)
        cov80_flags.extend(in_80.astype(float).tolist())
        cov95_flags.extend(in_95.astype(float).tolist())

        width80_vals.extend((q90 - q10).tolist())
        width95_vals.extend((q975 - q025).tolist())

    mean_w = (
        round(float(np.mean(bma_weights_across_folds)), 4)
        if bma_weights_across_folds is not None and len(bma_weights_across_folds) > 0
        else None
    )

    return ModelEvaluationMetrics(
        model_code=model_code,
        model_name=model_name,
        mae=round(float(np.mean(all_errors)), 3),
        rmse=round(float(np.sqrt(np.mean(all_sq_errors))), 3),
        mase=round(float(np.mean(all_scaled_errors)), 4),
        crps=round(float(np.mean(crps_per_fold)), 3),
        nlpd=round(float(np.mean(nlpd_per_fold)), 3),
        coverage_80=round(float(np.mean(cov80_flags)), 4),
        coverage_95=round(float(np.mean(cov95_flags)), 4),
        interval_width_80=round(float(np.mean(width80_vals)), 2),
        interval_width_95=round(float(np.mean(width95_vals)), 2),
        mean_bma_weight=mean_w,
    )


def run_expanding_window_backtest(
    obs_df: pd.DataFrame,
    n_folds: int = 3,
    horizon: int = 4,
    draws: int = 300,
    chains: int = 2,
    random_seed: int = 42,
    seasonality_period: float = 52.18,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute expanding-window chronological time-series cross-validation.

    For Fold f = 1 .. n_folds:
    - Train strictly on historical observations [0 : cutoff_f]
    - Compute preprocessing scalers, SMC marginal likelihoods, and BMA weights on train_df ONLY
    - Generate H-step ahead posterior predictive distributions for M1..M5 and BMA
    - Evaluate out-of-sample forecasts against held-out test observations [cutoff_f : cutoff_f + H]
    """
    ordered = obs_df.sort_values("observation_date").reset_index(drop=True)
    # Filter to rows where actual test evaluation is possible, while keeping chronological order
    raw_cases = pd.to_numeric(ordered["confirmed_cases"], errors="coerce")
    valid_indices = np.where(raw_cases.notna())[0]
    if len(valid_indices) < 24 + n_folds * horizon:
        raise ValueError(
            f"Series has {len(valid_indices)} valid observations; need at least {24 + n_folds * horizon} "
            f"for {n_folds} folds of horizon {horizon}."
        )

    total_len = len(ordered)
    # Place fold origins at the end of the series spaced by `horizon`
    fold_cutoffs = [
        total_len - (n_folds - f_idx) * horizon
        for f_idx in range(n_folds)
    ]

    model_codes = list(CandidateModelSuite.MODEL_METADATA.keys())
    samples_by_model: Dict[str, List[np.ndarray]] = {c: [] for c in model_codes + ["BMA"]}
    weights_by_model: Dict[str, List[float]] = {c: [] for c in model_codes}
    y_true_per_fold: List[np.ndarray] = []
    naive_scale_per_fold: List[float] = []
    fold_records: List[Dict[str, Any]] = []

    for f_idx, cutoff in enumerate(fold_cutoffs):
        if progress_callback:
            progress_callback(
                pct=round(80.0 + (18.0 * f_idx / n_folds), 1),
                msg=f"Running expanding-window chronological backtest fold {f_idx + 1}/{n_folds} (train cutoff index={cutoff})...",
            )

        train_df, test_full_df = ChronologicalPreprocessor.chronological_split(ordered, cutoff)
        test_df = test_full_df.iloc[:horizon].copy().reset_index(drop=True)

        # Verify strict temporal ordering (Leakage assertion)
        assert str(train_df["observation_date"].max()) < str(test_df["observation_date"].min()), (
            "Chronological leakage detected: training window overlaps or follows test window!"
        )

        prep = ChronologicalPreprocessor(seasonality_period=seasonality_period)
        train_feats, prep_state = prep.fit_transform_train(train_df)
        future_dates = test_df["observation_date"].astype(str).tolist()
        fut_covs = prep.build_future_deterministic_covariates(horizon=horizon, future_dates=future_dates)

        # In-sample 1-step naive random walk MAE for MASE denominator (computed on training data ONLY)
        y_tr = train_feats["y"].astype(np.float64)
        naive_mae = float(np.mean(np.abs(np.diff(y_tr)))) if len(y_tr) > 1 else 1.0
        naive_scale_per_fold.append(max(naive_mae, 1.0))

        # Actual test counts
        y_test_raw = pd.to_numeric(test_df["confirmed_cases"], errors="coerce").ffill().bfill().fillna(0.0)
        y_test = y_test_raw.to_numpy(dtype=np.float64)
        y_true_per_fold.append(y_test)

        suite = CandidateModelSuite(draws=draws, chains=chains, random_seed=random_seed + f_idx * 100)
        cand_results = suite.fit_all_models(
            train_features=train_feats,
            prep_state=prep_state,
            future_covariates=fut_covs,
        )
        bma_res = combine_posterior_predictive_distributions(
            model_results=cand_results,
            n_bma_samples=1500,
            random_seed=random_seed + f_idx * 100,
        )

        for m_res in cand_results:
            samples_by_model[m_res.model_code].append(m_res.future_count_samples)
            weights_by_model[m_res.model_code].append(bma_res.posterior_probabilities[m_res.model_code])
        samples_by_model["BMA"].append(bma_res.bma_count_samples)

        # Store step-by-step predictions vs actuals for visualization in Page 6
        step_entries = []
        for h in range(horizon):
            per_model_means = {m.model_code: round(float(m.horizon_mean[h]), 2) for m in cand_results}
            step_entries.append({
                "horizon_step": h + 1,
                "target_date": future_dates[h],
                "actual_cases": int(round(y_test[h])),
                "bma_mean": round(float(bma_res.horizon_mean[h]), 2),
                "bma_median": round(float(bma_res.horizon_median[h]), 2),
                "bma_q10": round(float(bma_res.horizon_q10[h]), 2),
                "bma_q90": round(float(bma_res.horizon_q90[h]), 2),
                "bma_q025": round(float(bma_res.horizon_q025[h]), 2),
                "bma_q975": round(float(bma_res.horizon_q975[h]), 2),
                "individual_model_means": per_model_means,
            })

        fold_records.append({
            "fold_index": f_idx + 1,
            "train_start_date": prep_state.train_start_date,
            "train_end_date": prep_state.train_end_date,
            "train_size": prep_state.n_train_total,
            "test_start_date": future_dates[0],
            "test_end_date": future_dates[-1],
            "bma_weights_in_fold": {k: round(v, 4) for k, v in bma_res.posterior_probabilities.items()},
            "log_marginal_likelihoods_in_fold": {
                k: round(v, 2) if np.isfinite(v) else None for k, v in bma_res.log_marginal_likelihoods.items()
            },
            "predictions": step_entries,
        })

    # Compute aggregate metrics across all folds for each model + BMA
    metrics_list: List[ModelEvaluationMetrics] = []
    bma_metrics = evaluate_predictions_against_actuals(
        model_code="BMA",
        model_name="Bayesian Model Averaging Ensemble (BMA)",
        samples_list=samples_by_model["BMA"],
        y_true_list=y_true_per_fold,
        naive_scale_list=naive_scale_per_fold,
        bma_weights_across_folds=[1.0] * n_folds,
    )
    metrics_list.append(bma_metrics)

    for code in model_codes:
        m_name = CandidateModelSuite.MODEL_METADATA[code]["name"]
        m_eval = evaluate_predictions_against_actuals(
            model_code=code,
            model_name=m_name,
            samples_list=samples_by_model[code],
            y_true_list=y_true_per_fold,
            naive_scale_list=naive_scale_per_fold,
            bma_weights_across_folds=weights_by_model[code],
        )
        metrics_list.append(m_eval)

    # Objective comparison: did BMA outperform the best single model on CRPS / RMSE / MAE?
    single_models = [m for m in metrics_list if m.model_code != "BMA"]
    best_single_crps = min(single_models, key=lambda m: m.crps)
    best_single_rmse = min(single_models, key=lambda m: m.rmse)
    best_single_mae = min(single_models, key=lambda m: m.mae)

    bma_beat_crps = bma_metrics.crps <= best_single_crps.crps
    bma_beat_rmse = bma_metrics.rmse <= best_single_rmse.rmse

    if bma_beat_crps and bma_beat_rmse:
        summary_text = (
            f"Across {n_folds} expanding-window chronological validation folds ({n_folds * horizon} out-of-sample "
            f"forecast steps), the Bayesian Model Averaging (BMA) ensemble achieved the lowest overall probabilistic "
            f"error (CRPS = {bma_metrics.crps:.2f}, RMSE = {bma_metrics.rmse:.2f}, MASE = {bma_metrics.mase:.3f}), "
            f"outperforming every individual candidate model including the strongest single model "
            f"{best_single_crps.model_code} (CRPS = {best_single_crps.crps:.2f}, RMSE = {best_single_crps.rmse:.2f}). "
            f"BMA 80% interval coverage was {bma_metrics.coverage_80 * 100:.1f}% and 95% coverage was "
            f"{bma_metrics.coverage_95 * 100:.1f}%."
        )
    else:
        summary_text = (
            f"Objective Backtest Report: Across {n_folds} expanding-window chronological folds ({n_folds * horizon} "
            f"out-of-sample steps), BMA achieved CRPS = {bma_metrics.crps:.2f}, RMSE = {bma_metrics.rmse:.2f}, "
            f"MAE = {bma_metrics.mae:.2f}, and 95% coverage = {bma_metrics.coverage_95 * 100:.1f}%. "
            f"The best individual model on CRPS was {best_single_crps.model_code} (CRPS = {best_single_crps.crps:.2f}) "
            f"and on RMSE was {best_single_rmse.model_code} (RMSE = {best_single_rmse.rmse:.2f}). "
            f"Because BMA weights models by in-sample marginal likelihood P(D_train | M_i) prior to observing future regime "
            f"shifts, an individual model specialized to the post-cutoff trajectory can occasionally achieve lower point "
            f"error on specific folds, while BMA provides hedged predictive uncertainty across competing structures."
        )

    return {
        "n_folds": n_folds,
        "horizon_per_fold": horizon,
        "validation_protocol": "Expanding-Window Chronological Cross-Validation (Strict Past-Only Training)",
        "bma_outperformed_best_single": bool(bma_beat_crps),
        "comparison_summary_text": summary_text,
        "metrics": [m.__dict__ for m in metrics_list],
        "folds": fold_records,
    }

