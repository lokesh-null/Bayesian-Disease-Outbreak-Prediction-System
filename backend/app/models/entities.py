"""Normalized SQLAlchemy ORM entities for the Disease Outbreak Prediction System."""
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class Disease(Base):
    __tablename__ = "diseases"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    default_seasonality_period = Column(Float, default=52.18, nullable=False)
    transmission_mode = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    observations = relationship("DiseaseObservation", back_populates="disease", cascade="all, delete-orphan")


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    region_type = Column(String(64), default="National/Regional")
    country = Column(String(128), default="United States")
    population = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    observations = relationship("DiseaseObservation", back_populates="location", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    source_name = Column(String(255), nullable=False)
    source_url = Column(String(512), nullable=True)
    attribution = Column(Text, nullable=True)
    downloaded_at = Column(DateTime, default=utcnow, nullable=False)
    file_hash = Column(String(128), nullable=False, index=True)
    row_count = Column(Integer, default=0, nullable=False)
    missing_count = Column(Integer, default=0, nullable=False)
    duplicate_count = Column(Integer, default=0, nullable=False)
    date_start = Column(String(32), nullable=True)
    date_end = Column(String(32), nullable=True)
    frequency = Column(String(32), default="WEEKLY", nullable=False)
    schema_mapping_json = Column(Text, nullable=True)

    observations = relationship("DiseaseObservation", back_populates="dataset", cascade="all, delete-orphan")


class DiseaseObservation(Base):
    __tablename__ = "disease_observations"
    __table_args__ = (
        UniqueConstraint("dataset_id", "disease_id", "location_id", "observation_date", name="uq_obs_series_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    disease_id = Column(Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False, index=True)
    location_id = Column(Integer, ForeignKey("locations.id", ondelete="CASCADE"), nullable=False, index=True)
    observation_date = Column(String(32), nullable=False, index=True)
    epiweek = Column(Integer, nullable=True, index=True)
    confirmed_cases = Column(Integer, nullable=True)
    deaths = Column(Integer, nullable=True)
    exposure_denominator = Column(Integer, nullable=True)
    population = Column(Integer, nullable=True)
    is_missing = Column(Boolean, default=False, nullable=False)
    is_imputed = Column(Boolean, default=False, nullable=False)
    raw_metadata_json = Column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="observations")
    disease = relationship("Disease", back_populates="observations")
    location = relationship("Location", back_populates="observations")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    disease_id = Column(Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False, index=True)
    location_id = Column(Integer, ForeignKey("locations.id", ondelete="CASCADE"), nullable=False, index=True)
    train_start_date = Column(String(32), nullable=False)
    train_end_date = Column(String(32), nullable=False)
    train_observations_count = Column(Integer, nullable=False)
    forecast_horizon = Column(Integer, default=8, nullable=False)
    random_seed = Column(Integer, default=42, nullable=False)
    sampling_config_json = Column(Text, nullable=False)
    status = Column(String(32), default="PENDING", nullable=False, index=True)
    progress_pct = Column(Float, default=0.0, nullable=False)
    status_message = Column(Text, nullable=True)
    model_version = Column(String(64), default="1.0.0", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    candidate_models = relationship("CandidateModelResult", back_populates="training_run", cascade="all, delete-orphan")
    forecasts = relationship("ForecastRecord", back_populates="training_run", cascade="all, delete-orphan")
    outbreak_assessments = relationship("OutbreakAssessment", back_populates="training_run", cascade="all, delete-orphan")
    validation_runs = relationship("ValidationRun", back_populates="training_run", cascade="all, delete-orphan")


class CandidateModelResult(Base):
    __tablename__ = "candidate_model_results"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_code = Column(String(64), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    model_family = Column(String(128), nullable=False)
    assumptions_text = Column(Text, nullable=False)
    converged = Column(Boolean, default=True, nullable=False)
    included_in_bma = Column(Boolean, default=True, nullable=False)
    exclusion_reason = Column(Text, nullable=True)
    log_marginal_likelihood = Column(Float, nullable=True)
    log_ml_std_error = Column(Float, nullable=True)
    laplace_log_evidence = Column(Float, nullable=True)
    waic = Column(Float, nullable=True)
    loo = Column(Float, nullable=True)
    prior_prob = Column(Float, nullable=False)
    posterior_prob = Column(Float, nullable=False)
    parameter_summary_json = Column(Text, nullable=True)
    in_sample_fitted_json = Column(Text, nullable=True)

    training_run = relationship("TrainingRun", back_populates="candidate_models")
    diagnostic = relationship("ModelDiagnostic", back_populates="candidate_model", uselist=False, cascade="all, delete-orphan")


class ModelDiagnostic(Base):
    __tablename__ = "model_diagnostics"

    id = Column(Integer, primary_key=True, index=True)
    candidate_model_id = Column(Integer, ForeignKey("candidate_model_results.id", ondelete="CASCADE"), nullable=False, unique=True)
    max_rhat = Column(Float, nullable=True)
    min_ess_bulk = Column(Float, nullable=True)
    min_ess_tail = Column(Float, nullable=True)
    divergences = Column(Integer, default=0, nullable=False)
    sampling_time_sec = Column(Float, nullable=True)
    smc_stages_json = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)

    candidate_model = relationship("CandidateModelResult", back_populates="diagnostic")


class ForecastRecord(Base):
    __tablename__ = "forecast_records"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_code = Column(String(64), nullable=False, index=True)  # 'BMA' or 'M1'..'M5'
    target_date = Column(String(32), nullable=False, index=True)
    epiweek = Column(Integer, nullable=True)
    horizon_step = Column(Integer, nullable=False)
    mean_pred = Column(Float, nullable=False)
    median_pred = Column(Float, nullable=False)
    std_pred = Column(Float, nullable=False)
    q025 = Column(Float, nullable=False)
    q10 = Column(Float, nullable=False)
    q25 = Column(Float, nullable=False)
    q75 = Column(Float, nullable=False)
    q90 = Column(Float, nullable=False)
    q975 = Column(Float, nullable=False)
    param_variance = Column(Float, default=0.0, nullable=False)
    obs_variance = Column(Float, default=0.0, nullable=False)
    model_variance = Column(Float, default=0.0, nullable=False)
    total_variance = Column(Float, default=0.0, nullable=False)
    predictive_samples_json = Column(Text, nullable=True)

    training_run = relationship("TrainingRun", back_populates="forecasts")


class OutbreakAssessment(Base):
    __tablename__ = "outbreak_assessments"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    target_date = Column(String(32), nullable=False)
    epiweek = Column(Integer, nullable=True)
    horizon_step = Column(Integer, nullable=False)
    seasonal_baseline_mean = Column(Float, nullable=False)
    seasonal_baseline_std = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    threshold_method = Column(String(128), nullable=False)
    exceedance_prob = Column(Float, nullable=False)
    expected_growth_rate = Column(Float, nullable=False)
    risk_level = Column(String(64), nullable=False)
    explanation_text = Column(Text, nullable=False)

    training_run = relationship("TrainingRun", back_populates="outbreak_assessments")


class ValidationRun(Base):
    __tablename__ = "validation_runs"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    n_folds = Column(Integer, nullable=False)
    horizon_per_fold = Column(Integer, nullable=False)
    validation_protocol = Column(String(128), default="Expanding-Window Chronological Cross-Validation")
    bma_outperformed_best_single = Column(Boolean, nullable=False)
    comparison_summary_text = Column(Text, nullable=False)
    fold_predictions_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    training_run = relationship("TrainingRun", back_populates="validation_runs")
    metrics = relationship("ValidationMetricRecord", back_populates="validation_run", cascade="all, delete-orphan")


class ValidationMetricRecord(Base):
    __tablename__ = "validation_metric_records"

    id = Column(Integer, primary_key=True, index=True)
    validation_run_id = Column(Integer, ForeignKey("validation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    model_code = Column(String(64), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mase = Column(Float, nullable=False)
    crps = Column(Float, nullable=False)
    nlpd = Column(Float, nullable=False)
    coverage_80 = Column(Float, nullable=False)
    coverage_95 = Column(Float, nullable=False)
    interval_width_80 = Column(Float, nullable=False)
    interval_width_95 = Column(Float, nullable=False)
    mean_bma_weight_across_folds = Column(Float, nullable=True)

    validation_run = relationship("ValidationRun", back_populates="metrics")

