"""
Outbreak Detection & Epidemiological Risk Classification Module.

Computes:
1. Analytical Seasonal Baseline Thresholds (Farrington-style endemic channel: moving +/-3 week
   historical window mean + z_alpha * std, or historical upper quantile threshold).
   Explicitly labeled as an analytical surveillance threshold rather than an official statutory
   public-health declaration when statutory thresholds are unavailable.
2. Posterior Predictive Outbreak Exceedance Probability:
   P(Y_{T+h} > T_outbreak(T+h) | D) = (1 / S_BMA) * sum_{s=1}^{S_BMA} I(y_{T+h}^{(s)} > T_outbreak(T+h))
3. Expected Relative Incidence Growth Rate:
   g_{T+h} = (E[Y_{T+h} | D] - y_T) / max(y_T, 1)
4. Configurable 4-Tier Risk Classification (Low, Moderate, High, Critical) with detailed
   scientific explanations for every horizon step.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class OutbreakHorizonAssessment:
    horizon_step: int
    target_date: str
    epiweek: Optional[int]
    seasonal_baseline_mean: float
    seasonal_baseline_std: float
    threshold_value: float
    threshold_method: str
    exceedance_prob: float
    expected_growth_rate: float
    bma_mean_cases: float
    bma_q95_upper: float
    risk_level: str
    explanation_text: str


class OutbreakRiskDetector:
    """
    Evaluates future BMA posterior predictive distributions against historical seasonal
    surveillance baselines learned strictly from the training window.
    """

    def __init__(
        self,
        z_multiplier: float = 1.28,  # ~90th percentile of Gaussian seasonal channel
        min_quantile_floor: float = 0.75,
        low_to_moderate_prob: float = 0.25,
        moderate_to_high_prob: float = 0.55,
        high_to_critical_prob: float = 0.80,
    ):
        self.z_multiplier = z_multiplier
        self.min_quantile_floor = min_quantile_floor
        self.low_to_moderate_prob = low_to_moderate_prob
        self.moderate_to_high_prob = moderate_to_high_prob
        self.high_to_critical_prob = high_to_critical_prob

    @staticmethod
    def _iso_week_from_date(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return min(int(dt.isocalendar()[1]), 52)

    def compute_seasonal_baseline_for_week(
        self, train_dates: List[str], train_y_valid: np.ndarray, target_iso_week: int
    ) -> Dict[str, float]:
        """
        Compute historical baseline mean, std, and analytical outbreak threshold for `target_iso_week`
        using a circular +/-3 week seasonal window across historical training observations.
        """
        train_weeks = np.array([self._iso_week_from_date(d) for d in train_dates], dtype=int)
        # Circular week distance on 1..52
        diff = np.abs(train_weeks - target_iso_week)
        circ_dist = np.minimum(diff, 52 - diff)
        window_mask = circ_dist <= 3

        window_vals = train_y_valid[window_mask]
        if len(window_vals) < 4:
            window_vals = train_y_valid

        base_mean = float(np.mean(window_vals))
        base_std = float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else float(np.std(train_y_valid))
        q_floor = float(np.quantile(window_vals, self.min_quantile_floor))

        # Analytical epidemic threshold: max(seasonal mean + z * seasonal std, historical 75th percentile, at least 1)
        threshold = max(base_mean + self.z_multiplier * base_std, q_floor, 1.0)
        return {
            "baseline_mean": round(base_mean, 2),
            "baseline_std": round(base_std, 2),
            "threshold": round(threshold, 2),
        }

    def classify_exceedance_probability(
        self, exceedance_prob: float, expected_growth_rate: float
    ) -> str:
        """Classify risk tier using posterior exceedance probability and growth dynamics."""
        if exceedance_prob >= self.high_to_critical_prob:
            return "Critical Risk"
        if exceedance_prob >= self.moderate_to_high_prob:
            return "High Risk"
        if exceedance_prob >= self.low_to_moderate_prob or (exceedance_prob >= 0.18 and expected_growth_rate > 0.35):
            return "Moderate Risk"
        return "Low Risk"

    def evaluate_forecast_risk(
        self,
        train_dates: List[str],
        train_y: np.ndarray,
        valid_mask: np.ndarray,
        future_dates: List[str],
        future_epiweeks: List[Optional[int]],
        bma_count_samples: np.ndarray,
        bma_posterior_weights: Dict[str, float],
    ) -> List[OutbreakHorizonAssessment]:
        """
        Evaluate outbreak exceedance probability P(Y_{T+h} > T_outbreak | D) across all forecast horizons.
        """
        valid_dates = [train_dates[i] for i in range(len(train_dates)) if valid_mask[i]]
        valid_y = train_y[valid_mask].astype(np.float64)
        last_observed_y = float(valid_y[-1]) if len(valid_y) > 0 else 1.0

        # Identify dominant models in BMA ensemble for explanation
        sorted_weights = sorted(bma_posterior_weights.items(), key=lambda kv: -kv[1])
        top_models_str = ", ".join(
            f"{code} ({w * 100:.1f}%)" for code, w in sorted_weights[:2] if w > 0.01
        )

        horizon = len(future_dates)
        assessments: List[OutbreakHorizonAssessment] = []

        for h in range(horizon):
            d_str = future_dates[h]
            ew = future_epiweeks[h] if future_epiweeks and h < len(future_epiweeks) else None
            iso_wk = self._iso_week_from_date(d_str)

            base_info = self.compute_seasonal_baseline_for_week(valid_dates, valid_y, iso_wk)
            thresh = base_info["threshold"]
            samples_h = bma_count_samples[:, h].astype(np.float64)

            exceed_prob = float(np.mean(samples_h > thresh))
            mean_h = float(np.mean(samples_h))
            q95_h = float(np.percentile(samples_h, 97.5))
            growth_rate = float((mean_h - last_observed_y) / max(last_observed_y, 1.0))

            risk_tier = self.classify_exceedance_probability(exceed_prob, growth_rate)
            method_label = (
                f"Analytical Seasonal Endemic Channel (ISO Week {iso_wk} ±3w historical mean + "
                f"{self.z_multiplier:.2f}σ / P{int(self.min_quantile_floor*100)} floor; non-statutory threshold)"
            )

            direction_str = (
                f"+{growth_rate * 100:.1f}% above" if growth_rate >= 0 else f"{growth_rate * 100:.1f}% below"
            )
            explanation = (
                f"Horizon +{h + 1}w ({d_str}): Classified as {risk_tier}. "
                f"The BMA posterior predictive probability of exceeding the analytical seasonal threshold "
                f"({thresh:.1f} cases; seasonal historical mean {base_info['baseline_mean']:.1f} ± {base_info['baseline_std']:.1f}) "
                f"is P(Y > T_outbreak | D) = {exceed_prob * 100:.1f}%. "
                f"Expected BMA incidence is {mean_h:.1f} cases (95% predictive upper bound: {q95_h:.1f}), "
                f"representing a {direction_str} change relative to the latest reported observation ({int(last_observed_y)} cases). "
                f"Ensemble signal is primarily weighted by {top_models_str}."
            )

            assessments.append(
                OutbreakHorizonAssessment(
                    horizon_step=h + 1,
                    target_date=d_str,
                    epiweek=ew,
                    seasonal_baseline_mean=base_info["baseline_mean"],
                    seasonal_baseline_std=base_info["baseline_std"],
                    threshold_value=thresh,
                    threshold_method=method_label,
                    exceedance_prob=round(exceed_prob, 4),
                    expected_growth_rate=round(growth_rate, 4),
                    bma_mean_cases=round(mean_h, 2),
                    bma_q95_upper=round(q95_h, 2),
                    risk_level=risk_tier,
                    explanation_text=explanation,
                )
            )

        return assessments

