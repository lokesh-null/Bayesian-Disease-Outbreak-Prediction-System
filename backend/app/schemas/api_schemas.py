"""Pydantic v2 request and response schemas for the BMA Outbreak Prediction API."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DiseaseSchema(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = None
    default_seasonality_period: float = 52.18
    transmission_mode: Optional[str] = None


class LocationSchema(BaseModel):
    id: int
    code: str
    name: str
    region_type: Optional[str] = None
    country: Optional[str] = None
    population: Optional[int] = None


class DatasetSchema(BaseModel):
    id: int
    name: str
    source_name: str
    source_url: Optional[str] = None
    attribution: Optional[str] = None
    downloaded_at: datetime
    file_hash: str
    row_count: int
    missing_count: int
    duplicate_count: int
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    frequency: str


class TrainForecastRequest(BaseModel):
    disease_code: str = Field(..., description="Disease code e.g. ILI, PERTUSSIS, DENGUE")
    location_code: str = Field(..., description="Location code e.g. US-NAT, US-HHS2, US-PR")
    forecast_horizon: int = Field(default=8, ge=1, le=26, description="Weeks ahead to forecast")
    smc_draws: int = Field(default=350, ge=100, le=2000, description="SMC particles/draws per chain")
    smc_chains: int = Field(default=2, ge=2, le=4, description="Number of independent SMC chains")
    random_seed: int = Field(default=42, description="Random seed for exact reproducibility")
    run_backtest: bool = Field(default=True, description="Whether to run expanding-window cross-validation")
    backtest_folds: int = Field(default=2, ge=1, le=6, description="Number of chronological backtest folds")
    backtest_horizon: int = Field(default=4, ge=1, le=12, description="Forecast horizon per backtest fold")
    outbreak_z_multiplier: float = Field(
        default=1.28, ge=0.5, le=4.0, description="Analytical seasonal threshold z-multiplier"
    )
    train_start_date: Optional[str] = Field(default=None, description="Optional ISO start date filter")
    train_end_date: Optional[str] = Field(default=None, description="Optional ISO end date filter")


class LiveFetchRequest(BaseModel):
    source_type: str = Field(default="CDC_ILINET", description="CDC_ILINET or CDC_NNDSS")
    regions: List[str] = Field(default_factory=lambda: ["nat", "hhs2", "hhs4"])
    epiweeks_range: str = Field(default="202201-202452")

