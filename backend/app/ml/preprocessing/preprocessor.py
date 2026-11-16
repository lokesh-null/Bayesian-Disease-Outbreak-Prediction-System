"""
Leakage-Safe Chronological Time-Series Preprocessing & Feature Engineering.

Strictly enforces:
- Rule 6: Never use future information when training models for historical validation.
- Rule 7: Never perform random train-test splitting on chronological disease surveillance data.
- Rule 9: Never silently ignore missing observations.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TemporalPreprocessorState:
    """Stores scaling and baseline parameters learned strictly on the training window."""
    train_start_date: str
    train_end_date: str
    n_train_total: int
    n_train_valid: int
    n_train_missing: int
    t_mean: float
    t_std: float
    y_train_mean: float
    y_train_std: float
    log_y_plus1_mean: float
    seasonality_period: float = 52.18
    serial_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    last_observed_counts: List[int] = field(default_factory=list)
    last_time_index: int = 0
    last_observation_date: str = ""


class ChronologicalPreprocessor:
    """
    Fit-on-train-only preprocessor for epidemiological count time series.
    Constructs temporal trend, Fourier harmonic terms, autoregressive lags,
    and renewal infectious load features without ever leaking future actuals.
    """

    def __init__(self, seasonality_period: float = 52.18, max_harmonics: int = 3):
        self.seasonality_period = seasonality_period
        self.max_harmonics = max_harmonics
        self.state: Optional[TemporalPreprocessorState] = None

    @staticmethod
    def chronological_split(
        df: pd.DataFrame, cutoff_date_or_index: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a chronological dataframe into strictly ordered (train_df, test_df).
        Never shuffles or randomly samples observations.
        """
        ordered = df.sort_values("observation_date").reset_index(drop=True)
        if isinstance(cutoff_date_or_index, int):
            if cutoff_date_or_index <= 0 or cutoff_date_or_index >= len(ordered):
                raise ValueError(f"Invalid split index {cutoff_date_or_index} for length {len(ordered)}")
            train_df = ordered.iloc[:cutoff_date_or_index].copy().reset_index(drop=True)
            test_df = ordered.iloc[cutoff_date_or_index:].copy().reset_index(drop=True)
        else:
            cutoff_str = str(cutoff_date_or_index)
            train_df = ordered[ordered["observation_date"] <= cutoff_str].copy().reset_index(drop=True)
            test_df = ordered[ordered["observation_date"] > cutoff_str].copy().reset_index(drop=True)
        return train_df, test_df

    def fit_transform_train(
        self, train_df: pd.DataFrame, missing_strategy: str = "interpolate_past_only"
    ) -> Tuple[Dict[str, Any], TemporalPreprocessorState]:
        """
        Learn scaling parameters strictly from `train_df` and build model design arrays.
        Missing observations are explicitly tracked via `is_missing_mask` and handled
        only using causal (backward/past-only) fill for autoregressive lag continuity.
        """
        if train_df is None or len(train_df) < 8:
            raise ValueError("Training time series must contain at least 8 chronological observations.")

        ordered = train_df.sort_values("observation_date").reset_index(drop=True)
        n_total = len(ordered)
        t_idx = np.arange(n_total, dtype=np.float64)

        raw_cases = pd.to_numeric(ordered["confirmed_cases"], errors="coerce")
        is_missing_mask = raw_cases.isna().to_numpy(dtype=bool)
        n_missing = int(is_missing_mask.sum())
        n_valid = n_total - n_missing

        if n_valid < 6:
            raise ValueError("Insufficient non-missing observations in training window.")

        # Causal (past-only) imputation for lag construction only; never uses future bfill across cutoff
        causal_filled = raw_cases.ffill()
        if causal_filled.isna().any():
            # Only if the very first observation(s) are missing, fill with the first valid training observation
            first_valid = float(raw_cases.dropna().iloc[0])
            causal_filled = causal_filled.fillna(first_valid)

        y_clean = np.maximum(np.round(causal_filled.to_numpy(dtype=np.float64)), 0).astype(np.int64)

        t_mean = float(np.mean(t_idx))
        t_std = float(np.std(t_idx)) if np.std(t_idx) > 1e-8 else 1.0
        t_scaled = (t_idx - t_mean) / t_std

        valid_vals = raw_cases.dropna().to_numpy(dtype=np.float64)
        y_train_mean = float(np.mean(valid_vals))
        y_train_std = float(np.std(valid_vals)) if np.std(valid_vals) > 1e-8 else 1.0
        log_y_plus1_mean = float(np.mean(np.log(valid_vals + 1.0)))

        # Fourier harmonic features (k = 1..max_harmonics)
        fourier_sin = []
        fourier_cos = []
        for k in range(1, self.max_harmonics + 1):
            angle = 2.0 * np.pi * k * t_idx / self.seasonality_period
            fourier_sin.append(np.sin(angle))
            fourier_cos.append(np.cos(angle))
        fourier_sin_arr = np.column_stack(fourier_sin)  # shape (N, K)
        fourier_cos_arr = np.column_stack(fourier_cos)  # shape (N, K)

        # Causal lagged log-counts: lag 1 and lag 2 (strictly historical)
        log_y = np.log(y_clean.astype(np.float64) + 1.0)
        lag1_log_y = np.empty_like(log_y)
        lag2_log_y = np.empty_like(log_y)
        lag1_log_y[0] = log_y[0]
        lag1_log_y[1:] = log_y[:-1]
        lag2_log_y[0] = log_y[0]
        lag2_log_y[1] = log_y[0]
        lag2_log_y[2:] = log_y[:-2]

        lag1_centered = lag1_log_y - log_y_plus1_mean
        lag2_centered = lag2_log_y - log_y_plus1_mean

        # Renewal infectious load I_{t-1} = sum_{tau=1..3} w_tau * y_{t-tau} + 1.0
        w = [0.5, 0.3, 0.2]
        infectious_load = np.empty(n_total, dtype=np.float64)
        for t in range(n_total):
            y_m1 = y_clean[t - 1] if t >= 1 else y_clean[0]
            y_m2 = y_clean[t - 2] if t >= 2 else y_m1
            y_m3 = y_clean[t - 3] if t >= 3 else y_m2
            infectious_load[t] = w[0] * y_m1 + w[1] * y_m2 + w[2] * y_m3 + 1.0
        log_infectious_load = np.log(infectious_load)
        log_infectious_load_centered = log_infectious_load - log_y_plus1_mean

        self.state = TemporalPreprocessorState(
            train_start_date=str(ordered["observation_date"].iloc[0]),
            train_end_date=str(ordered["observation_date"].iloc[-1]),
            n_train_total=n_total,
            n_train_valid=n_valid,
            n_train_missing=n_missing,
            t_mean=t_mean,
            t_std=t_std,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            log_y_plus1_mean=log_y_plus1_mean,
            seasonality_period=self.seasonality_period,
            serial_weights=w,
            last_observed_counts=[int(x) for x in y_clean[-3:]],
            last_time_index=n_total - 1,
            last_observation_date=str(ordered["observation_date"].iloc[-1]),
        )

        features = {
            "y": y_clean,
            "is_missing_mask": is_missing_mask,
            "valid_mask": ~is_missing_mask,
            "t_idx": t_idx,
            "t_scaled": t_scaled,
            "fourier_sin": fourier_sin_arr,
            "fourier_cos": fourier_cos_arr,
            "lag1_centered": lag1_centered,
            "lag2_centered": lag2_centered,
            "log_infectious_load": log_infectious_load,
            "log_infectious_load_centered": log_infectious_load_centered,
            "dates": ordered["observation_date"].astype(str).tolist(),
            "epiweeks": [
                int(x) if pd.notna(x) else None for x in ordered.get("epiweek", pd.Series([None] * n_total))
            ],
        }
        return features, self.state

    def build_future_deterministic_covariates(
        self, horizon: int, future_dates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate deterministic future covariates (time trend and seasonal Fourier harmonics)
        for steps T+1 .. T+H using ONLY the training preprocessor state.
        Note: Autoregressive and renewal features are NOT pre-filled from future observations;
        they are rolled out dynamically inside each model's posterior predictive generator.
        """
        if self.state is None:
            raise RuntimeError("Preprocessor must be fitted on training data before generating future covariates.")
        if horizon < 1:
            raise ValueError("Forecast horizon must be >= 1.")

        future_t_idx = np.arange(
            self.state.last_time_index + 1,
            self.state.last_time_index + 1 + horizon,
            dtype=np.float64,
        )
        future_t_scaled = (future_t_idx - self.state.t_mean) / self.state.t_std

        fourier_sin = []
        fourier_cos = []
        for k in range(1, self.max_harmonics + 1):
            angle = 2.0 * np.pi * k * future_t_idx / self.state.seasonality_period
            fourier_sin.append(np.sin(angle))
            fourier_cos.append(np.cos(angle))

        if future_dates is None:
            last_dt = datetime.strptime(self.state.last_observation_date, "%Y-%m-%d")
            future_dates = [
                (last_dt + timedelta(days=7 * h)).strftime("%Y-%m-%d")
                for h in range(1, horizon + 1)
            ]

        future_epiweeks = []
        for d_str in future_dates:
            dt_obj = datetime.strptime(d_str, "%Y-%m-%d")
            iso_c = dt_obj.isocalendar()
            future_epiweeks.append(int(iso_c[0] * 100 + iso_c[1]))

        return {
            "horizon": horizon,
            "t_idx": future_t_idx,
            "t_scaled": future_t_scaled,
            "fourier_sin": np.column_stack(fourier_sin),
            "fourier_cos": np.column_stack(fourier_cos),
            "dates": future_dates,
            "epiweeks": future_epiweeks,
        }

