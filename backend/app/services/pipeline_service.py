"""
End-to-End Pipeline Orchestrator Service:
Coordinates data retrieval, chronological preprocessing, PyMC SMC training of all 5 candidate models,
BMA posterior mixture combination, outbreak exceedance risk classification, expanding-window
chronological validation, and full database persistence.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import DATA_DIR
from app.database.session import SessionLocal
from app.ml.bayesian_models.candidate_models import CandidateModelSuite
from app.ml.data_ingestion.ingestor import ingest_file_bytes
from app.ml.evaluation.backtester import run_expanding_window_backtest
from app.ml.model_averaging.bma_engine import combine_posterior_predictive_distributions
from app.ml.outbreak_detection.detector import OutbreakRiskDetector
from app.ml.preprocessing.preprocessor import ChronologicalPreprocessor
from app.models.entities import (
    CandidateModelResult,
    Dataset,
    Disease,
    DiseaseObservation,
    ForecastRecord,
    Location,
    ModelDiagnostic,
    OutbreakAssessment,
    TrainingRun,
    ValidationMetricRecord,
    ValidationRun,
)


def get_observations_dataframe(
    db: Session,
    disease_code: str,
    location_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Query chronological disease observations from SQLAlchemy database."""
    disease = db.query(Disease).filter(Disease.code == disease_code.upper()).first()
    location = db.query(Location).filter(Location.code == location_code.upper()).first()
    if not disease or not location:
        raise ValueError(f"Disease '{disease_code}' or Location '{location_code}' not found in database.")

    q = (
        db.query(DiseaseObservation)
        .filter(
            DiseaseObservation.disease_id == disease.id,
            DiseaseObservation.location_id == location.id,
        )
    )
    if start_date:
        q = q.filter(DiseaseObservation.observation_date >= start_date)
    if end_date:
        q = q.filter(DiseaseObservation.observation_date <= end_date)

    rows = q.order_by(DiseaseObservation.observation_date.asc()).all()
    if not rows:
        raise ValueError(f"No observations found for {disease_code} in {location_code}.")

    records = []
    for r in rows:
        records.append({
            "id": r.id,
            "dataset_id": r.dataset_id,
            "observation_date": r.observation_date,
            "epiweek": r.epiweek,
            "disease_code": disease.code,
            "disease_name": disease.name,
            "location_code": location.code,
            "location_name": location.name,
            "confirmed_cases": r.confirmed_cases,
            "deaths": r.deaths,
            "exposure_denominator": r.exposure_denominator,
            "population": r.population or location.population,
            "is_missing": r.is_missing,
            "is_imputed": r.is_imputed,
        })
    df = pd.DataFrame(records)
    # Deduplicate if multiple datasets overlap on same date
    df = df.drop_duplicates(subset=["observation_date"], keep="last").sort_values("observation_date").reset_index(drop=True)
    return df


def execute_training_run_pipeline(
    training_run_id: int,
    run_backtest: bool = True,
    backtest_folds: int = 2,
    backtest_horizon: int = 4,
    outbreak_z_multiplier: float = 1.28,
) -> None:
    """
    Execute the complete Bayesian training, BMA synthesis, outbreak risk estimation,
    and chronological validation pipeline for a `TrainingRun` record.
    Can be invoked synchronously or in a background worker thread.
    """
    db = SessionLocal()
    try:
        tr = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if not tr:
            return

        tr.status = "RUNNING"
        tr.progress_pct = 5.0
        tr.status_message = "Loading and preprocessing chronological surveillance series..."
        db.commit()

        disease = db.query(Disease).filter(Disease.id == tr.disease_id).first()
        location = db.query(Location).filter(Location.id == tr.location_id).first()
        cfg = json.loads(tr.sampling_config_json)
        smc_draws = int(cfg.get("smc_draws", 300))
        smc_chains = int(cfg.get("smc_chains", 2))

        obs_df = get_observations_dataframe(
            db=db,
            disease_code=disease.code,
            location_code=location.code,
            start_date=tr.train_start_date if tr.train_start_date != "ALL" else None,
            end_date=tr.train_end_date if tr.train_end_date != "ALL" else None,
        )

        # Keep most recent 180 weeks if series is very long so SMC sampling stays fast and responsive
        if len(obs_df) > 180 and tr.train_start_date == "ALL":
            obs_df = obs_df.iloc[-180:].reset_index(drop=True)

        seasonality_period = float(disease.default_seasonality_period or 52.18)
        prep = ChronologicalPreprocessor(seasonality_period=seasonality_period)
        train_feats, prep_state = prep.fit_transform_train(obs_df)
        future_covs = prep.build_future_deterministic_covariates(horizon=tr.forecast_horizon)

        tr.train_start_date = prep_state.train_start_date
        tr.train_end_date = prep_state.train_end_date
        tr.train_observations_count = prep_state.n_train_total
        db.commit()

        def _update_progress(pct: float, msg: str):
            tr.progress_pct = pct
            tr.status_message = msg
            db.commit()

        suite = CandidateModelSuite(
            draws=smc_draws,
            chains=smc_chains,
            random_seed=tr.random_seed,
        )
        cand_results = suite.fit_all_models(
            train_features=train_feats,
            prep_state=prep_state,
            future_covariates=future_covs,
            progress_callback=_update_progress,
        )

        _update_progress(76.0, "Combining posterior predictive distributions via Bayesian Model Averaging...")
        bma_res = combine_posterior_predictive_distributions(
            model_results=cand_results,
            n_bma_samples=2000,
            random_seed=tr.random_seed,
        )

        # Persist candidate model results and ArviZ diagnostics
        in_sample_dates = train_feats["dates"]
        in_sample_actuals = [
            None if bool(train_feats["is_missing_mask"][i]) else int(train_feats["y"][i])
            for i in range(len(in_sample_dates))
        ]

        for m_res in cand_results:
            # Sample 60 evenly spaced in-sample points or last 80 points for compact JSON storage
            tail_n = min(104, len(in_sample_dates))
            in_sample_payload = [
                {
                    "date": in_sample_dates[-tail_n + i],
                    "actual": in_sample_actuals[-tail_n + i],
                    "mean": round(float(m_res.in_sample_mean[-tail_n + i]), 2),
                    "q10": round(float(m_res.in_sample_q10[-tail_n + i]), 2),
                    "q90": round(float(m_res.in_sample_q90[-tail_n + i]), 2),
                }
                for i in range(tail_n)
            ]

            cm_db = CandidateModelResult(
                training_run_id=tr.id,
                model_code=m_res.model_code,
                model_name=m_res.model_name,
                model_family=m_res.model_family,
                assumptions_text=m_res.assumptions_text,
                converged=m_res.converged,
                included_in_bma=bool(bma_res.posterior_probabilities.get(m_res.model_code, 0.0) > 0.0 or m_res.converged),
                exclusion_reason=m_res.exclusion_reason,
                log_marginal_likelihood=float(m_res.log_marginal_likelihood) if np.isfinite(m_res.log_marginal_likelihood) else None,
                log_ml_std_error=float(m_res.log_ml_std_error) if np.isfinite(m_res.log_ml_std_error) else None,
                laplace_log_evidence=float(m_res.laplace_log_evidence) if np.isfinite(m_res.laplace_log_evidence) else None,
                waic=float(m_res.waic),
                loo=float(m_res.loo),
                prior_prob=float(bma_res.prior_probabilities[m_res.model_code]),
                posterior_prob=float(bma_res.posterior_probabilities[m_res.model_code]),
                parameter_summary_json=json.dumps(m_res.parameter_summary),
                in_sample_fitted_json=json.dumps(in_sample_payload),
            )
            db.add(cm_db)
            db.flush()

            diag_db = ModelDiagnostic(
                candidate_model_id=cm_db.id,
                max_rhat=float(m_res.max_rhat),
                min_ess_bulk=float(m_res.min_ess_bulk),
                min_ess_tail=float(m_res.min_ess_tail),
                divergences=int(m_res.divergences),
                sampling_time_sec=float(m_res.sampling_time_sec),
                smc_stages_json=json.dumps({"chain_log_marginal_likelihoods": m_res.smc_chain_log_mls}),
                warnings_json=json.dumps(m_res.warnings_list),
            )
            db.add(diag_db)

            # Persist individual model future forecast records
            for h in range(tr.forecast_horizon):
                hist_counts, hist_edges = np.histogram(m_res.future_count_samples[:, h], bins=12)
                dist_summary = [
                    {
                        "bin_center": round(float(0.5 * (hist_edges[b] + hist_edges[b + 1])), 1),
                        "density": round(float(hist_counts[b] / len(m_res.future_count_samples[:, h])), 4),
                    }
                    for b in range(len(hist_counts))
                ]
                db.add(
                    ForecastRecord(
                        training_run_id=tr.id,
                        model_code=m_res.model_code,
                        target_date=future_covs["dates"][h],
                        epiweek=future_covs["epiweeks"][h],
                        horizon_step=h + 1,
                        mean_pred=round(float(m_res.horizon_mean[h]), 2),
                        median_pred=round(float(m_res.horizon_median[h]), 2),
                        std_pred=round(float(m_res.horizon_std[h]), 2),
                        q025=round(float(m_res.horizon_q025[h]), 2),
                        q10=round(float(m_res.horizon_q10[h]), 2),
                        q25=round(float(m_res.horizon_q25[h]), 2),
                        q75=round(float(m_res.horizon_q75[h]), 2),
                        q90=round(float(m_res.horizon_q90[h]), 2),
                        q975=round(float(m_res.horizon_q975[h]), 2),
                        param_variance=round(float(m_res.param_variance[h]), 3),
                        obs_variance=round(float(m_res.obs_variance[h]), 3),
                        model_variance=0.0,
                        total_variance=round(float(m_res.total_variance[h]), 3),
                        predictive_samples_json=json.dumps(dist_summary),
                    )
                )

        # Persist BMA ensemble future forecast records with full 3-way uncertainty decomposition
        for h in range(tr.forecast_horizon):
            bma_draws_h = bma_res.bma_count_samples[:, h]
            hist_counts, hist_edges = np.histogram(bma_draws_h, bins=16)
            dist_summary = [
                {
                    "bin_center": round(float(0.5 * (hist_edges[b] + hist_edges[b + 1])), 1),
                    "count": int(hist_counts[b]),
                    "probability": round(float(hist_counts[b] / len(bma_draws_h)), 4),
                }
                for b in range(len(hist_counts))
            ]
            db.add(
                ForecastRecord(
                    training_run_id=tr.id,
                    model_code="BMA",
                    target_date=future_covs["dates"][h],
                    epiweek=future_covs["epiweeks"][h],
                    horizon_step=h + 1,
                    mean_pred=round(float(bma_res.horizon_mean[h]), 2),
                    median_pred=round(float(bma_res.horizon_median[h]), 2),
                    std_pred=round(float(bma_res.horizon_std[h]), 2),
                    q025=round(float(bma_res.horizon_q025[h]), 2),
                    q10=round(float(bma_res.horizon_q10[h]), 2),
                    q25=round(float(bma_res.horizon_q25[h]), 2),
                    q75=round(float(bma_res.horizon_q75[h]), 2),
                    q90=round(float(bma_res.horizon_q90[h]), 2),
                    q975=round(float(bma_res.horizon_q975[h]), 2),
                    param_variance=round(float(bma_res.within_param_variance[h]), 3),
                    obs_variance=round(float(bma_res.within_obs_variance[h]), 3),
                    model_variance=round(float(bma_res.between_model_variance[h]), 3),
                    total_variance=round(float(bma_res.total_predictive_variance[h]), 3),
                    predictive_samples_json=json.dumps(dist_summary),
                )
            )

        # Outbreak detection & risk classification
        _update_progress(80.0, "Evaluating outbreak exceedance probabilities and risk classifications...")
        detector = OutbreakRiskDetector(z_multiplier=outbreak_z_multiplier)
        assessments = detector.evaluate_forecast_risk(
            train_dates=train_feats["dates"],
            train_y=train_feats["y"],
            valid_mask=train_feats["valid_mask"],
            future_dates=future_covs["dates"],
            future_epiweeks=future_covs["epiweeks"],
            bma_count_samples=bma_res.bma_count_samples,
            bma_posterior_weights=bma_res.posterior_probabilities,
        )
        for a in assessments:
            db.add(
                OutbreakAssessment(
                    training_run_id=tr.id,
                    target_date=a.target_date,
                    epiweek=a.epiweek,
                    horizon_step=a.horizon_step,
                    seasonal_baseline_mean=a.seasonal_baseline_mean,
                    seasonal_baseline_std=a.seasonal_baseline_std,
                    threshold_value=a.threshold_value,
                    threshold_method=a.threshold_method,
                    exceedance_prob=a.exceedance_prob,
                    expected_growth_rate=a.expected_growth_rate,
                    risk_level=a.risk_level,
                    explanation_text=a.explanation_text,
                )
            )
        db.commit()

        # Expanding-window chronological validation
        if run_backtest and len(obs_df) >= 32:
            bt_draws = max(180, smc_draws // 2)
            bt_res = run_expanding_window_backtest(
                obs_df=obs_df,
                n_folds=backtest_folds,
                horizon=backtest_horizon,
                draws=bt_draws,
                chains=smc_chains,
                random_seed=tr.random_seed,
                seasonality_period=seasonality_period,
                progress_callback=_update_progress,
            )
            val_run = ValidationRun(
                training_run_id=tr.id,
                n_folds=bt_res["n_folds"],
                horizon_per_fold=bt_res["horizon_per_fold"],
                validation_protocol=bt_res["validation_protocol"],
                bma_outperformed_best_single=bt_res["bma_outperformed_best_single"],
                comparison_summary_text=bt_res["comparison_summary_text"],
                fold_predictions_json=json.dumps(bt_res["folds"]),
            )
            db.add(val_run)
            db.flush()

            for m_dict in bt_res["metrics"]:
                db.add(
                    ValidationMetricRecord(
                        validation_run_id=val_run.id,
                        model_code=m_dict["model_code"],
                        model_name=m_dict["model_name"],
                        mae=m_dict["mae"],
                        rmse=m_dict["rmse"],
                        mase=m_dict["mase"],
                        crps=m_dict["crps"],
                        nlpd=m_dict["nlpd"],
                        coverage_80=m_dict["coverage_80"],
                        coverage_95=m_dict["coverage_95"],
                        interval_width_80=m_dict["interval_width_80"],
                        interval_width_95=m_dict["interval_width_95"],
                        mean_bma_weight_across_folds=m_dict["mean_bma_weight"],
                    )
                )

        tr.status = "COMPLETED"
        tr.progress_pct = 100.0
        tr.status_message = "Bayesian Model Averaging training, forecasting, and chronological validation completed."
        tr.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.rollback()
        tr = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if tr:
            tr.status = "FAILED"
            tr.status_message = f"Training failed: {str(exc)}\n{traceback.format_exc()[:500]}"
            db.commit()
        raise
    finally:
        db.close()


def ensure_initial_datasets_and_baseline_runs(db: Session) -> None:
    """
    Ensure the two genuine canonical surveillance datasets (CDC ILINet & CDC NNDSS)
    are ingested into the database on startup.
    """
    ilinet_csv = DATA_DIR / "cdc_ilinet_weekly_2017_2024.csv"
    nndss_csv = DATA_DIR / "cdc_nndss_weekly_2022_2024.csv"

    if ilinet_csv.exists():
        ingest_file_bytes(
            db=db,
            file_bytes=ilinet_csv.read_bytes(),
            filename=ilinet_csv.name,
            dataset_name="CDC ILINet Weekly Influenza-Like Illness Surveillance (2017–2024)",
            source_name="U.S. CDC Outpatient Influenza-like Illness Surveillance Network (ILINet) via CMU Delphi Epidata API",
            source_url="https://api.delphi.cmu.edu/epidata/fluview/",
            attribution="Centers for Disease Control and Prevention (CDC) / CMU Delphi Research Group. Public Domain U.S. Government Surveillance Data.",
        )

    if nndss_csv.exists():
        ingest_file_bytes(
            db=db,
            file_bytes=nndss_csv.read_bytes(),
            filename=nndss_csv.name,
            dataset_name="CDC NNDSS Weekly Notifiable Infectious Diseases (Dengue, Pertussis, Salmonellosis, Campylobacter, Malaria)",
            source_name="U.S. CDC National Notifiable Diseases Surveillance System (NNDSS) Socrata API",
            source_url="https://data.cdc.gov/resource/x9gk-5huc.json",
            attribution="Centers for Disease Control and Prevention (CDC) NNDSS Weekly Provisional Tables. Public Domain.",
        )

