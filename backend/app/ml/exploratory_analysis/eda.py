"""
Comprehensive Exploratory Data Analysis (EDA) Module for Epidemiological Surveillance Data.

Computes:
1. Summary & dispersion statistics (mean, variance, variance-to-mean ratio / index of dispersion)
2. Weekly and monthly incidence trends
3. Cumulative case trajectories & week-over-week growth rates
4. Rolling averages (4-week and 12-week moving averages)
5. Classical seasonal decomposition (trend, seasonal, residual via statsmodels)
6. Seasonal profile by epidemiological week (1..52) with interquartile bands
7. Outlier and sudden surge anomaly detection (modified Z-score & rolling 3-sigma spikes)
8. Missing observation audit
9. Multi-region geographic comparison & cross-variable correlation analysis
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def perform_exploratory_analysis(
    obs_df: pd.DataFrame,
    all_locations_df: Optional[pd.DataFrame] = None,
    seasonality_period: int = 52,
) -> Dict[str, Any]:
    """
    Execute comprehensive epidemiological EDA on a chronological surveillance series.
    Clearly distinguishes reported observations from missing records and derived indicators.
    """
    if obs_df is None or obs_df.empty:
        raise ValueError("Cannot run EDA on an empty observation series.")

    df = obs_df.sort_values("observation_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(df["observation_date"])
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["year_month"] = dates.dt.strftime("%Y-%m")
    iso_week = dates.dt.isocalendar()["week"].astype(int)
    df["week_of_year"] = np.minimum(iso_week, 52)

    raw_cases = pd.to_numeric(df["confirmed_cases"], errors="coerce")
    is_missing = raw_cases.isna() | df.get("is_missing", pd.Series([False] * len(df))).astype(bool)
    valid_cases = raw_cases[~is_missing].to_numpy(dtype=np.float64)

    if len(valid_cases) == 0:
        raise ValueError("All observations in the selected series are missing.")

    mean_val = float(np.mean(valid_cases))
    std_val = float(np.std(valid_cases))
    var_val = float(np.var(valid_cases))
    median_val = float(np.median(valid_cases))
    iqr_val = float(np.percentile(valid_cases, 75) - np.percentile(valid_cases, 25))
    min_val = float(np.min(valid_cases))
    max_val = float(np.max(valid_cases))
    # Index of Dispersion (Variance-to-Mean Ratio, VMR): >1 indicates overdispersion relative to Poisson
    vmr = float(var_val / max(mean_val, 1e-8))
    zero_count = int(np.sum(valid_cases == 0))
    zero_pct = float(100.0 * zero_count / len(valid_cases))

    # Causal rolling metrics & growth rates
    filled_causal = raw_cases.ffill().bfill().fillna(0.0)
    ma_4w = filled_causal.rolling(window=4, min_periods=1).mean()
    ma_12w = filled_causal.rolling(window=12, min_periods=1).mean()
    rolling_std_8w = filled_causal.rolling(window=8, min_periods=3).std().fillna(std_val)
    cumulative_cases = raw_cases.fillna(0).cumsum()

    prev_cases = filled_causal.shift(1)
    growth_rate_pct = np.where(
        prev_cases > 0,
        ((filled_causal - prev_cases) / prev_cases) * 100.0,
        0.0,
    )

    # Seasonal decomposition (trend, seasonal, residual)
    decomp_period = min(seasonality_period, max(4, len(df) // 2))
    if len(df) >= 2 * decomp_period:
        decomp = seasonal_decompose(
            filled_causal.to_numpy(dtype=np.float64),
            model="additive",
            period=decomp_period,
            extrapolate_trend="period",
        )
        trend_comp = decomp.trend
        seasonal_comp = decomp.seasonal
        resid_comp = decomp.resid
    else:
        trend_comp = ma_12w.to_numpy(dtype=np.float64)
        seasonal_comp = (filled_causal - ma_12w).to_numpy(dtype=np.float64)
        resid_comp = np.zeros(len(df), dtype=np.float64)

    # Outlier & Sudden Surge Detection
    # Sudden surge: reported cases exceed 4-week moving average by > 2.0 rolling stds and > median
    resid_std = float(np.std(resid_comp)) if np.std(resid_comp) > 1e-8 else 1.0
    z_scores = (filled_causal.to_numpy(dtype=np.float64) - ma_4w.to_numpy(dtype=np.float64)) / np.maximum(
        rolling_std_8w.to_numpy(dtype=np.float64), 1.0
    )

    time_series_records: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []
    missing_dates: List[str] = []

    for idx in range(len(df)):
        d_str = str(df["observation_date"].iloc[idx])
        miss_flag = bool(is_missing.iloc[idx])
        if miss_flag:
            missing_dates.append(d_str)
        c_val = None if miss_flag else int(raw_cases.iloc[idx])
        z_val = float(z_scores[idx])
        is_outlier = (not miss_flag) and (abs(resid_comp[idx]) > 2.5 * resid_std)
        is_sudden_surge = (
            (not miss_flag)
            and (c_val is not None)
            and (z_val > 2.0)
            and (c_val > mean_val)
        )

        rec = {
            "observation_date": d_str,
            "epiweek": int(df["epiweek"].iloc[idx]) if pd.notna(df["epiweek"].iloc[idx]) else None,
            "reported_cases": c_val,
            "is_missing": miss_flag,
            "ma_4w": round(float(ma_4w.iloc[idx]), 2),
            "ma_12w": round(float(ma_12w.iloc[idx]), 2),
            "cumulative_cases": int(cumulative_cases.iloc[idx]),
            "growth_rate_pct": round(float(growth_rate_pct[idx]), 2),
            "trend_component": round(float(trend_comp[idx]), 2),
            "seasonal_component": round(float(seasonal_comp[idx]), 2),
            "residual_component": round(float(resid_comp[idx]), 2),
            "z_score": round(z_val, 2),
            "is_outlier": bool(is_outlier),
            "is_sudden_surge": bool(is_sudden_surge),
        }
        time_series_records.append(rec)

        if is_outlier or is_sudden_surge:
            anomalies.append({
                "observation_date": d_str,
                "reported_cases": c_val,
                "ma_4w": round(float(ma_4w.iloc[idx]), 2),
                "z_score": round(z_val, 2),
                "type": "Sudden Incidence Surge" if is_sudden_surge else "Statistical Residual Outlier",
            })

    # Monthly trend aggregation
    df_valid = df[~is_missing].copy()
    df_valid["cases_num"] = raw_cases[~is_missing]
    monthly_trends = []
    for ym, grp in df_valid.groupby("year_month"):
        monthly_trends.append({
            "year_month": str(ym),
            "total_cases": int(grp["cases_num"].sum()),
            "mean_weekly_cases": round(float(grp["cases_num"].mean()), 2),
            "max_weekly_cases": int(grp["cases_num"].max()),
            "weeks_reported": int(len(grp)),
        })

    # Seasonal profile by week of year (1..52)
    seasonal_profile = []
    for wk, grp in df_valid.groupby("week_of_year"):
        vals = grp["cases_num"].to_numpy(dtype=np.float64)
        seasonal_profile.append({
            "week_of_year": int(wk),
            "mean_cases": round(float(np.mean(vals)), 2),
            "median_cases": round(float(np.median(vals)), 2),
            "q25_cases": round(float(np.percentile(vals, 25)), 2),
            "q75_cases": round(float(np.percentile(vals, 75)), 2),
            "q90_cases": round(float(np.percentile(vals, 90)), 2),
        })

    # Distribution histogram bins for case counts
    hist_counts, hist_edges = np.histogram(valid_cases, bins=min(15, max(5, len(valid_cases) // 10)))
    distribution_bins = []
    for b_i in range(len(hist_counts)):
        distribution_bins.append({
            "bin_start": round(float(hist_edges[b_i]), 1),
            "bin_end": round(float(hist_edges[b_i + 1]), 1),
            "bin_label": f"{int(round(hist_edges[b_i]))}–{int(round(hist_edges[b_i + 1]))}",
            "frequency": int(hist_counts[b_i]),
        })

    # Correlation analysis across explanatory & temporal variables
    corr_df = pd.DataFrame({
        "confirmed_cases": filled_causal,
        "lag_1w_cases": filled_causal.shift(1).bfill(),
        "lag_2w_cases": filled_causal.shift(2).bfill(),
        "ma_4w": ma_4w,
        "seasonal_sin": np.sin(2.0 * np.pi * df["week_of_year"] / 52.18),
        "seasonal_cos": np.cos(2.0 * np.pi * df["week_of_year"] / 52.18),
    })
    if "exposure_denominator" in df.columns and df["exposure_denominator"].notna().sum() > 5:
        corr_df["exposure_denominator"] = pd.to_numeric(df["exposure_denominator"], errors="coerce").ffill().bfill()

    corr_matrix = corr_df.corr().fillna(0.0)
    correlations = []
    for var_name in corr_matrix.columns:
        if var_name == "confirmed_cases":
            continue
        correlations.append({
            "variable": var_name,
            "pearson_r_with_cases": round(float(corr_matrix.loc["confirmed_cases", var_name]), 4),
        })

    # Geographic comparison if multi-location dataframe provided
    geographic_comparison = []
    if all_locations_df is not None and not all_locations_df.empty and "location_code" in all_locations_df.columns:
        for l_code, grp in all_locations_df.groupby("location_code"):
            c_s = pd.to_numeric(grp["confirmed_cases"], errors="coerce").dropna()
            if c_s.empty:
                continue
            l_name = str(grp["location_name"].iloc[0]) if "location_name" in grp.columns else str(l_code)
            pop_s = pd.to_numeric(grp.get("population", pd.Series(dtype=float)), errors="coerce").dropna()
            pop_val = int(pop_s.iloc[0]) if not pop_s.empty else None
            total_c = int(c_s.sum())
            mean_c = float(c_s.mean())
            incidence_per_100k = round((mean_c / pop_val) * 100000.0, 2) if pop_val else None
            geographic_comparison.append({
                "location_code": str(l_code),
                "location_name": l_name,
                "observations": int(len(c_s)),
                "total_cases": total_c,
                "mean_weekly_cases": round(mean_c, 2),
                "max_weekly_cases": int(c_s.max()),
                "population": pop_val,
                "mean_incidence_per_100k": incidence_per_100k,
            })

    return {
        "summary_statistics": {
            "total_observations": int(len(df)),
            "valid_observations": int(len(valid_cases)),
            "missing_observations": int(is_missing.sum()),
            "missing_pct": round(float(100.0 * is_missing.sum() / len(df)), 2),
            "date_start": str(df["observation_date"].iloc[0]),
            "date_end": str(df["observation_date"].iloc[-1]),
            "mean_cases": round(mean_val, 2),
            "std_cases": round(std_val, 2),
            "variance_cases": round(var_val, 2),
            "variance_to_mean_ratio": round(vmr, 2),
            "overdispersed": bool(vmr > 1.5),
            "median_cases": round(median_val, 2),
            "iqr_cases": round(iqr_val, 2),
            "min_cases": int(min_val),
            "max_cases": int(max_val),
            "total_cumulative_cases": int( np.sum(valid_cases) ),
            "zero_count": zero_count,
            "zero_pct": round(zero_pct, 2),
            "latest_observation_date": str(df["observation_date"].iloc[-1]),
            "latest_reported_cases": int(valid_cases[-1]),
        },
        "missing_dates": missing_dates,
        "anomalies_and_surges": anomalies[-30:],  # most recent 30 surges/outliers
        "total_anomalies_detected": len(anomalies),
        "time_series": time_series_records,
        "monthly_trends": monthly_trends,
        "seasonal_profile": seasonal_profile,
        "distribution_bins": distribution_bins,
        "correlations": correlations,
        "geographic_comparison": geographic_comparison,
    }
