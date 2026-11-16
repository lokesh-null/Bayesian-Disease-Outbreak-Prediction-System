"""
FastAPI REST API Endpoints for the Disease Outbreak Prediction System Using Bayesian Model Averaging.

Provides endpoints for:
- Dataset retrieval, CSV/Excel upload, and live CMU Delphi / CDC API sync
- Disease and Geographic Location selection
- Exploratory Data Analysis (EDA) & anomaly inspection
- Bayesian Model Training (synchronous or background job execution) & job status polling
- Candidate Model Analysis, ArviZ diagnostics, and BMA posterior weights
- Outbreak Forecast & Risk Assessment
- 3-Way Uncertainty Decomposition (Parameter + Observation + Structural Model uncertainty)
- Expanding-Window Chronological Cross-Validation & Proper Scoring Rules evaluation
- Overview dashboard summary
"""
import json
from typing import Any, Dict, List, Optional
import urllib.request
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.ml.data_ingestion.ingestor import ingest_file_bytes
from app.ml.exploratory_analysis.eda import perform_exploratory_analysis
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
from app.schemas.api_schemas import LiveFetchRequest, TrainForecastRequest
from app.services.pipeline_service import (
    execute_training_run_pipeline,
    get_observations_dataframe,
)

router = APIRouter()


def _get_latest_completed_run(
    db: Session,
    disease_code: Optional[str] = None,
    location_code: Optional[str] = None,
    run_id: Optional[int] = None,
) -> Optional[TrainingRun]:
    if run_id is not None:
        return db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    q = db.query(TrainingRun).filter(TrainingRun.status == "COMPLETED")
    if disease_code:
        d = db.query(Disease).filter(Disease.code == disease_code.upper()).first()
        if d:
            q = q.filter(TrainingRun.disease_id == d.id)
    if location_code:
        loc = db.query(Location).filter(Location.code == location_code.upper()).first()
        if loc:
            q = q.filter(TrainingRun.location_id == loc.id)

    run = q.order_by(TrainingRun.id.desc()).first()
    if not run:
        run = db.query(TrainingRun).filter(TrainingRun.status == "COMPLETED").order_by(TrainingRun.id.desc()).first()
    return run


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    datasets = db.query(Dataset).order_by(Dataset.id.asc()).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "source_name": d.source_name,
            "source_url": d.source_url,
            "attribution": d.attribution,
            "downloaded_at": d.downloaded_at.isoformat() if d.downloaded_at else None,
            "file_hash": d.file_hash,
            "row_count": d.row_count,
            "missing_count": d.missing_count,
            "duplicate_count": d.duplicate_count,
            "date_start": d.date_start,
            "date_end": d.date_end,
            "frequency": d.frequency,
        }
        for d in datasets
    ]


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(default=None),
    source_name: str = Form(default="User Uploaded Surveillance Dataset"),
    attribution: str = Form(default="Uploaded via Data Explorer"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    content = await file.read()
    try:
        ds, report = ingest_file_bytes(
            db=db,
            file_bytes=content,
            filename=file.filename,
            dataset_name=dataset_name or file.filename,
            source_name=source_name,
            attribution=attribution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "dataset": {
            "id": ds.id,
            "name": ds.name,
            "row_count": ds.row_count,
            "missing_count": ds.missing_count,
            "duplicate_count": ds.duplicate_count,
            "date_start": ds.date_start,
            "date_end": ds.date_end,
            "file_hash": ds.file_hash,
        },
        "ingestion_report": report,
    }


@router.get("/diseases")
def list_diseases(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    diseases = db.query(Disease).order_by(Disease.code.asc()).all()
    out = []
    for d in diseases:
        # Find locations that actually have observations for this disease
        loc_ids = (
            db.query(DiseaseObservation.location_id)
            .filter(DiseaseObservation.disease_id == d.id)
            .distinct()
            .all()
        )
        loc_id_list = [r[0] for r in loc_ids]
        locs = db.query(Location).filter(Location.id.in_(loc_id_list)).order_by(Location.code.asc()).all() if loc_id_list else []
        out.append({
            "id": d.id,
            "code": d.code,
            "name": d.name,
            "description": d.description,
            "default_seasonality_period": d.default_seasonality_period,
            "available_locations": [
                {"code": l.code, "name": l.name, "population": l.population}
                for l in locs
            ],
        })
    return out


@router.get("/locations")
def list_locations(
    disease_code: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    if disease_code:
        d = db.query(Disease).filter(Disease.code == disease_code.upper()).first()
        if not d:
            raise HTTPException(status_code=404, detail=f"Disease '{disease_code}' not found.")
        loc_ids = [
            r[0]
            for r in db.query(DiseaseObservation.location_id)
            .filter(DiseaseObservation.disease_id == d.id)
            .distinct()
            .all()
        ]
        locs = db.query(Location).filter(Location.id.in_(loc_ids)).order_by(Location.code.asc()).all()
    else:
        locs = db.query(Location).order_by(Location.code.asc()).all()

    return [
        {
            "id": l.id,
            "code": l.code,
            "name": l.name,
            "region_type": l.region_type,
            "country": l.country,
            "population": l.population,
        }
        for l in locs
    ]


@router.get("/observations")
def get_observations(
    disease_code: str = Query(default="ILI"),
    location_code: str = Query(default="US-NAT"),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    limit: int = Query(default=600, le=5000),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        df = get_observations_dataframe(db, disease_code, location_code, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    tail_df = df.iloc[-limit:].copy()
    records = []
    for row in tail_df.to_dict(orient="records"):
        records.append({
            "observation_date": row["observation_date"],
            "epiweek": int(row["epiweek"]) if row["epiweek"] is not None and str(row["epiweek"]) != "<NA>" else None,
            "confirmed_cases": None if (row["confirmed_cases"] is None or bool(row["is_missing"])) else int(row["confirmed_cases"]),
            "exposure_denominator": None if row["exposure_denominator"] is None else int(row["exposure_denominator"]),
            "population": None if row["population"] is None else int(row["population"]),
            "is_missing": bool(row["is_missing"]),
        })

    return {
        "disease_code": disease_code.upper(),
        "location_code": location_code.upper(),
        "total_matching_observations": len(df),
        "returned_observations": len(records),
        "observations": records,
    }


@router.get("/eda")
def get_eda_analysis(
    disease_code: str = Query(default="ILI"),
    location_code: str = Query(default="US-NAT"),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        obs_df = get_observations_dataframe(db, disease_code, location_code, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Also retrieve all locations for this disease for geographic comparison
    disease = db.query(Disease).filter(Disease.code == disease_code.upper()).first()
    all_loc_rows = (
        db.query(DiseaseObservation, Location)
        .join(Location, DiseaseObservation.location_id == Location.id)
        .filter(DiseaseObservation.disease_id == disease.id)
        .all()
    )
    geo_records = [
        {
            "location_code": loc.code,
            "location_name": loc.name,
            "confirmed_cases": obs.confirmed_cases,
            "population": obs.population or loc.population,
        }
        for obs, loc in all_loc_rows
    ]
    all_loc_df = pd.DataFrame(geo_records) if geo_records else None

    eda_payload = perform_exploratory_analysis(
        obs_df=obs_df,
        all_locations_df=all_loc_df,
        seasonality_period=int(round(disease.default_seasonality_period or 52)),
    )
    eda_payload["disease"] = {"code": disease.code, "name": disease.name}
    eda_payload["location_code"] = location_code.upper()
    return eda_payload


@router.post("/training/run")
def trigger_training_run(
    req: TrainForecastRequest,
    background_tasks: BackgroundTasks,
    sync: bool = Query(default=False, description="Run synchronously for immediate response or in background"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    disease = db.query(Disease).filter(Disease.code == req.disease_code.upper()).first()
    location = db.query(Location).filter(Location.code == req.location_code.upper()).first()
    if not disease or not location:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{req.disease_code}' or Location '{req.location_code}' not found.",
        )

    # Verify observations exist
    obs_count = (
        db.query(DiseaseObservation)
        .filter(
            DiseaseObservation.disease_id == disease.id,
            DiseaseObservation.location_id == location.id,
        )
        .count()
    )
    if obs_count < 12:
        raise HTTPException(
            status_code=400,
            detail=f"Series ({req.disease_code}, {req.location_code}) has only {obs_count} observations; at least 12 are required.",
        )

    tr = TrainingRun(
        disease_id=disease.id,
        location_id=location.id,
        train_start_date=req.train_start_date or "ALL",
        train_end_date=req.train_end_date or "ALL",
        train_observations_count=obs_count,
        forecast_horizon=req.forecast_horizon,
        random_seed=req.random_seed,
        sampling_config_json=json.dumps({
            "smc_draws": req.smc_draws,
            "smc_chains": req.smc_chains,
            "run_backtest": req.run_backtest,
            "backtest_folds": req.backtest_folds,
            "backtest_horizon": req.backtest_horizon,
            "outbreak_z_multiplier": req.outbreak_z_multiplier,
        }),
        status="PENDING",
        progress_pct=0.0,
        status_message="Queued for Bayesian SMC model estimation...",
    )
    db.add(tr)
    db.commit()
    db.refresh(tr)

    if sync:
        execute_training_run_pipeline(
            training_run_id=tr.id,
            run_backtest=req.run_backtest,
            backtest_folds=req.backtest_folds,
            backtest_horizon=req.backtest_horizon,
            outbreak_z_multiplier=req.outbreak_z_multiplier,
        )
        db.refresh(tr)
    else:
        background_tasks.add_task(
            execute_training_run_pipeline,
            tr.id,
            req.run_backtest,
            req.backtest_folds,
            req.backtest_horizon,
            req.outbreak_z_multiplier,
        )

    return {
        "training_run_id": tr.id,
        "disease_code": disease.code,
        "location_code": location.code,
        "status": tr.status,
        "progress_pct": tr.progress_pct,
        "status_message": tr.status_message,
    }


@router.get("/training/runs")
def list_training_runs(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    q = db.query(TrainingRun)
    if disease_code:
        d = db.query(Disease).filter(Disease.code == disease_code.upper()).first()
        if d:
            q = q.filter(TrainingRun.disease_id == d.id)
    if location_code:
        loc = db.query(Location).filter(Location.code == location_code.upper()).first()
        if loc:
            q = q.filter(TrainingRun.location_id == loc.id)

    runs = q.order_by(TrainingRun.id.desc()).limit(25).all()
    out = []
    for r in runs:
        d = db.query(Disease).filter(Disease.id == r.disease_id).first()
        loc = db.query(Location).filter(Location.id == r.location_id).first()
        out.append({
            "id": r.id,
            "disease_code": d.code if d else None,
            "disease_name": d.name if d else None,
            "location_code": loc.code if loc else None,
            "location_name": loc.name if loc else None,
            "train_start_date": r.train_start_date,
            "train_end_date": r.train_end_date,
            "train_observations_count": r.train_observations_count,
            "forecast_horizon": r.forecast_horizon,
            "random_seed": r.random_seed,
            "status": r.status,
            "progress_pct": r.progress_pct,
            "status_message": r.status_message,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        })
    return out


@router.get("/training/runs/{run_id}/status")
def get_training_run_status(run_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    r = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found.")
    return {
        "id": r.id,
        "status": r.status,
        "progress_pct": r.progress_pct,
        "status_message": r.status_message,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
    }


@router.get("/models/analysis")
def get_model_analysis(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    run_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    run = _get_latest_completed_run(db, disease_code, location_code, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No completed Bayesian training runs found.")

    disease = db.query(Disease).filter(Disease.id == run.disease_id).first()
    location = db.query(Location).filter(Location.id == run.location_id).first()
    cands = (
        db.query(CandidateModelResult)
        .filter(CandidateModelResult.training_run_id == run.id)
        .order_by(CandidateModelResult.posterior_prob.desc())
        .all()
    )

    models_out = []
    for c in cands:
        diag = db.query(ModelDiagnostic).filter(ModelDiagnostic.candidate_model_id == c.id).first()
        smc_info = json.loads(diag.smc_stages_json) if (diag and diag.smc_stages_json) else {}
        models_out.append({
            "model_code": c.model_code,
            "model_name": c.model_name,
            "model_family": c.model_family,
            "assumptions_text": c.assumptions_text,
            "converged": c.converged,
            "included_in_bma": c.included_in_bma,
            "exclusion_reason": c.exclusion_reason,
            "log_marginal_likelihood": round(c.log_marginal_likelihood, 2) if c.log_marginal_likelihood is not None else None,
            "log_ml_std_error": round(c.log_ml_std_error, 3) if c.log_ml_std_error is not None else None,
            "laplace_log_evidence": round(c.laplace_log_evidence, 2) if c.laplace_log_evidence is not None else None,
            "waic": round(c.waic, 2) if c.waic is not None else None,
            "loo": round(c.loo, 2) if c.loo is not None else None,
            "prior_prob": round(c.prior_prob, 4),
            "posterior_prob": round(c.posterior_prob, 4),
            "bma_weight_pct": round(c.posterior_prob * 100.0, 2),
            "diagnostics": {
                "max_rhat": round(diag.max_rhat, 4) if diag and diag.max_rhat is not None else None,
                "min_ess_bulk": round(diag.min_ess_bulk, 1) if diag and diag.min_ess_bulk is not None else None,
                "min_ess_tail": round(diag.min_ess_tail, 1) if diag and diag.min_ess_tail is not None else None,
                "divergences": diag.divergences if diag else 0,
                "sampling_time_sec": diag.sampling_time_sec if diag else None,
                "chain_log_marginal_likelihoods": smc_info.get("chain_log_marginal_likelihoods", []),
                "warnings": json.loads(diag.warnings_json) if (diag and diag.warnings_json) else [],
            },
            "parameters": json.loads(c.parameter_summary_json) if c.parameter_summary_json else [],
            "in_sample_fitted": json.loads(c.in_sample_fitted_json) if c.in_sample_fitted_json else [],
        })

    return {
        "training_run_id": run.id,
        "disease": {"code": disease.code, "name": disease.name},
        "location": {"code": location.code, "name": location.name},
        "train_start_date": run.train_start_date,
        "train_end_date": run.train_end_date,
        "train_observations_count": run.train_observations_count,
        "forecast_horizon": run.forecast_horizon,
        "random_seed": run.random_seed,
        "sampling_config": json.loads(run.sampling_config_json),
        "bma_weight_sum": round(sum(m["posterior_prob"] for m in models_out), 6),
        "candidate_models": models_out,
    }


@router.get("/forecasts")
def get_forecasts_and_risk(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    run_id: Optional[int] = Query(default=None),
    history_weeks: int = Query(default=52, ge=12, le=260),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    run = _get_latest_completed_run(db, disease_code, location_code, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No completed Bayesian forecast runs found.")

    disease = db.query(Disease).filter(Disease.id == run.disease_id).first()
    location = db.query(Location).filter(Location.id == run.location_id).first()

    # Historical observations up to train_end_date
    hist_df = get_observations_dataframe(
        db=db,
        disease_code=disease.code,
        location_code=location.code,
        end_date=run.train_end_date,
    ).iloc[-history_weeks:]

    historical_series = [
        {
            "observation_date": str(row["observation_date"]),
            "epiweek": int(row["epiweek"]) if row["epiweek"] is not None and str(row["epiweek"]) != "<NA>" else None,
            "reported_cases": None if (row["confirmed_cases"] is None or bool(row["is_missing"])) else int(row["confirmed_cases"]),
            "is_missing": bool(row["is_missing"]),
        }
        for row in hist_df.to_dict(orient="records")
    ]

    f_rows = (
        db.query(ForecastRecord)
        .filter(ForecastRecord.training_run_id == run.id)
        .order_by(ForecastRecord.horizon_step.asc())
        .all()
    )
    risk_rows = (
        db.query(OutbreakAssessment)
        .filter(OutbreakAssessment.training_run_id == run.id)
        .order_by(OutbreakAssessment.horizon_step.asc())
        .all()
    )
    cands = (
        db.query(CandidateModelResult)
        .filter(CandidateModelResult.training_run_id == run.id)
        .all()
    )
    bma_weights = {c.model_code: round(c.posterior_prob, 4) for c in cands}

    bma_forecasts = []
    individual_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for f in f_rows:
        item = {
            "horizon_step": f.horizon_step,
            "target_date": f.target_date,
            "epiweek": f.epiweek,
            "mean": f.mean_pred,
            "median": f.median_pred,
            "std": f.std_pred,
            "q025": f.q025,
            "q10": f.q10,
            "q25": f.q25,
            "q75": f.q75,
            "q90": f.q90,
            "q975": f.q975,
            "param_variance": f.param_variance,
            "obs_variance": f.obs_variance,
            "model_variance": f.model_variance,
            "total_variance": f.total_variance,
            "predictive_distribution": json.loads(f.predictive_samples_json) if f.predictive_samples_json else [],
        }
        if f.model_code == "BMA":
            bma_forecasts.append(item)
        else:
            individual_by_model.setdefault(f.model_code, []).append(item)

    risk_assessments = [
        {
            "horizon_step": r.horizon_step,
            "target_date": r.target_date,
            "epiweek": r.epiweek,
            "seasonal_baseline_mean": r.seasonal_baseline_mean,
            "seasonal_baseline_std": r.seasonal_baseline_std,
            "threshold_value": r.threshold_value,
            "threshold_method": r.threshold_method,
            "exceedance_prob": r.exceedance_prob,
            "exceedance_prob_pct": round(r.exceedance_prob * 100.0, 1),
            "expected_growth_rate": r.expected_growth_rate,
            "expected_growth_pct": round(r.expected_growth_rate * 100.0, 1),
            "risk_level": r.risk_level,
            "explanation_text": r.explanation_text,
        }
        for r in risk_rows
    ]

    # Combine step-by-step table for easy Recharts rendering
    combined_horizon_table = []
    for idx, bma_f in enumerate(bma_forecasts):
        h_step = bma_f["horizon_step"]
        r_obj = risk_assessments[idx] if idx < len(risk_assessments) else None
        row_dict: Dict[str, Any] = {
            "horizon_step": h_step,
            "target_date": bma_f["target_date"],
            "epiweek": bma_f["epiweek"],
            "bma_mean": bma_f["mean"],
            "bma_median": bma_f["median"],
            "bma_q10": bma_f["q10"],
            "bma_q90": bma_f["q90"],
            "bma_q025": bma_f["q025"],
            "bma_q975": bma_f["q975"],
            "threshold_value": r_obj["threshold_value"] if r_obj else None,
            "exceedance_prob_pct": r_obj["exceedance_prob_pct"] if r_obj else None,
            "risk_level": r_obj["risk_level"] if r_obj else "Low Risk",
        }
        for m_code, m_list in individual_by_model.items():
            if idx < len(m_list):
                row_dict[f"{m_code}_mean"] = m_list[idx]["mean"]
                row_dict[f"{m_code}_q10"] = m_list[idx]["q10"]
                row_dict[f"{m_code}_q90"] = m_list[idx]["q90"]
        combined_horizon_table.append(row_dict)

    peak_risk = max(risk_assessments, key=lambda x: x["exceedance_prob"]) if risk_assessments else None

    return {
        "training_run_id": run.id,
        "disease": {"code": disease.code, "name": disease.name},
        "location": {"code": location.code, "name": location.name, "population": location.population},
        "train_start_date": run.train_start_date,
        "train_end_date": run.train_end_date,
        "forecast_horizon": run.forecast_horizon,
        "bma_weights": bma_weights,
        "historical_observations": historical_series,
        "bma_forecasts": bma_forecasts,
        "individual_model_forecasts": individual_by_model,
        "outbreak_risk_assessments": risk_assessments,
        "combined_horizon_table": combined_horizon_table,
        "peak_risk_summary": peak_risk,
    }


@router.get("/uncertainty")
def get_uncertainty_decomposition(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    run_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Provide detailed Law of Total Variance uncertainty decomposition (Parameter Uncertainty,
    Aleatoric Observation Noise, and Structural Between-Model Uncertainty) across forecast horizons.
    """
    run = _get_latest_completed_run(db, disease_code, location_code, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No completed Bayesian training runs found.")

    disease = db.query(Disease).filter(Disease.id == run.disease_id).first()
    location = db.query(Location).filter(Location.id == run.location_id).first()

    bma_rows = (
        db.query(ForecastRecord)
        .filter(
            ForecastRecord.training_run_id == run.id,
            ForecastRecord.model_code == "BMA",
        )
        .order_by(ForecastRecord.horizon_step.asc())
        .all()
    )
    cands = (
        db.query(CandidateModelResult)
        .filter(CandidateModelResult.training_run_id == run.id)
        .order_by(CandidateModelResult.posterior_prob.desc())
        .all()
    )

    horizon_decomposition = []
    for f in bma_rows:
        # Components: param_variance, obs_variance, model_variance
        p_var = max(float(f.param_variance), 0.0)
        o_var = max(float(f.obs_variance), 0.0)
        m_var = max(float(f.model_variance), 0.0)
        sum_comp = max(p_var + o_var + m_var, 1e-8)

        horizon_decomposition.append({
            "horizon_step": f.horizon_step,
            "target_date": f.target_date,
            "mean": f.mean_pred,
            "median": f.median_pred,
            "interval_80": [f.q10, f.q90],
            "interval_95": [f.q025, f.q975],
            "width_80": round(f.q90 - f.q10, 2),
            "width_95": round(f.q975 - f.q025, 2),
            "parameter_variance": round(p_var, 2),
            "observation_variance": round(o_var, 2),
            "structural_model_variance": round(m_var, 2),
            "total_variance": round(float(f.total_variance), 2),
            "parameter_share_pct": round(100.0 * p_var / sum_comp, 1),
            "observation_share_pct": round(100.0 * o_var / sum_comp, 1),
            "model_share_pct": round(100.0 * m_var / sum_comp, 1),
            "predictive_distribution": json.loads(f.predictive_samples_json) if f.predictive_samples_json else [],
        })

    # Posterior model probability entropy (Shannon entropy H(M | D) = - sum w_i log(w_i))
    weights = [c.posterior_prob for c in cands if c.posterior_prob > 1e-12]
    import math
    shannon_entropy = -sum(w * math.log(w) for w in weights) if weights else 0.0
    max_entropy = math.log(max(len(cands), 1))
    normalized_model_entropy = round(shannon_entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    return {
        "training_run_id": run.id,
        "disease": {"code": disease.code, "name": disease.name},
        "location": {"code": location.code, "name": location.name},
        "model_space_entropy": round(shannon_entropy, 4),
        "normalized_model_entropy": normalized_model_entropy,
        "posterior_model_distribution": [
            {
                "model_code": c.model_code,
                "model_name": c.model_name,
                "prior_prob": round(c.prior_prob, 4),
                "posterior_prob": round(c.posterior_prob, 4),
                "log_marginal_likelihood": round(c.log_marginal_likelihood, 2) if c.log_marginal_likelihood is not None else None,
                "converged": c.converged,
            }
            for c in cands
        ],
        "horizon_uncertainty_decomposition": horizon_decomposition,
    }


@router.get("/evaluation")
def get_model_evaluation(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    run_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    run = _get_latest_completed_run(db, disease_code, location_code, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No completed Bayesian training runs found.")

    disease = db.query(Disease).filter(Disease.id == run.disease_id).first()
    location = db.query(Location).filter(Location.id == run.location_id).first()

    val_run = (
        db.query(ValidationRun)
        .filter(ValidationRun.training_run_id == run.id)
        .order_by(ValidationRun.id.desc())
        .first()
    )
    if not val_run:
        return {
            "training_run_id": run.id,
            "disease": {"code": disease.code, "name": disease.name},
            "location": {"code": location.code, "name": location.name},
            "has_validation": False,
            "metrics": [],
            "folds": [],
        }

    metrics = (
        db.query(ValidationMetricRecord)
        .filter(ValidationMetricRecord.validation_run_id == val_run.id)
        .all()
    )

    return {
        "training_run_id": run.id,
        "validation_run_id": val_run.id,
        "disease": {"code": disease.code, "name": disease.name},
        "location": {"code": location.code, "name": location.name},
        "has_validation": True,
        "n_folds": val_run.n_folds,
        "horizon_per_fold": val_run.horizon_per_fold,
        "validation_protocol": val_run.validation_protocol,
        "bma_outperformed_best_single": val_run.bma_outperformed_best_single,
        "comparison_summary_text": val_run.comparison_summary_text,
        "metrics": [
            {
                "model_code": m.model_code,
                "model_name": m.model_name,
                "mae": m.mae,
                "rmse": m.rmse,
                "mase": m.mase,
                "crps": m.crps,
                "nlpd": m.nlpd,
                "coverage_80": m.coverage_80,
                "coverage_80_pct": round(m.coverage_80 * 100.0, 1),
                "coverage_95": m.coverage_95,
                "coverage_95_pct": round(m.coverage_95 * 100.0, 1),
                "interval_width_80": m.interval_width_80,
                "interval_width_95": m.interval_width_95,
                "mean_bma_weight_across_folds": m.mean_bma_weight_across_folds,
            }
            for m in metrics
        ],
        "folds": json.loads(val_run.fold_predictions_json) if val_run.fold_predictions_json else [],
    }


@router.get("/overview")
def get_overview_dashboard(
    disease_code: Optional[str] = Query(default=None),
    location_code: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Unified payload for Page 1: Overview."""
    run = _get_latest_completed_run(db, disease_code, location_code)
    if not run:
        raise HTTPException(status_code=404, detail="No completed training runs available.")

    disease = db.query(Disease).filter(Disease.id == run.disease_id).first()
    location = db.query(Location).filter(Location.id == run.location_id).first()
    dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first() if run.dataset_id else db.query(Dataset).first()

    forecasts_payload = get_forecasts_and_risk(
        disease_code=disease.code,
        location_code=location.code,
        run_id=run.id,
        history_weeks=52,
        db=db,
    )
    eval_payload = get_model_evaluation(
        disease_code=disease.code,
        location_code=location.code,
        run_id=run.id,
        db=db,
    )

    hist = forecasts_payload["historical_observations"]
    valid_hist = [h for h in hist if h["reported_cases"] is not None]
    latest_obs = valid_hist[-1] if valid_hist else {"observation_date": run.train_end_date, "reported_cases": 0}
    prev_obs = valid_hist[-2] if len(valid_hist) >= 2 else latest_obs
    wow_change_pct = (
        round(100.0 * (latest_obs["reported_cases"] - prev_obs["reported_cases"]) / max(prev_obs["reported_cases"], 1), 1)
        if valid_hist
        else 0.0
    )

    return {
        "training_run_id": run.id,
        "disease": {"code": disease.code, "name": disease.name, "description": disease.description},
        "location": {"code": location.code, "name": location.name, "population": location.population},
        "data_freshness": {
            "latest_observation_date": latest_obs["observation_date"],
            "latest_reported_cases": latest_obs["reported_cases"],
            "week_over_week_change_pct": wow_change_pct,
            "source_name": dataset.source_name if dataset else "CDC Surveillance Repository",
            "source_url": dataset.source_url if dataset else None,
            "last_trained_at": run.completed_at.isoformat() if run.completed_at else None,
        },
        "bma_weights": forecasts_payload["bma_weights"],
        "peak_risk_summary": forecasts_payload["peak_risk_summary"],
        "historical_observations": hist,
        "bma_forecasts": forecasts_payload["bma_forecasts"],
        "combined_horizon_table": forecasts_payload["combined_horizon_table"],
        "outbreak_risk_assessments": forecasts_payload["outbreak_risk_assessments"],
        "evaluation_summary": {
            "has_validation": eval_payload["has_validation"],
            "bma_outperformed_best_single": eval_payload.get("bma_outperformed_best_single"),
            "comparison_summary_text": eval_payload.get("comparison_summary_text"),
            "metrics": eval_payload.get("metrics", []),
        },
    }
