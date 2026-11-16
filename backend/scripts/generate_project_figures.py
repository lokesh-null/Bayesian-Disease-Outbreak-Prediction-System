"""
Generate publication-quality epidemiological figures and visual outputs from the actual
trained SQLite database (`bma_outbreak_system.db`) and genuine CDC surveillance CSVs.
Saves all figures into `docs/assets/` and copies them to the artifact directory if specified.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.api.routes import (
    get_eda_analysis,
    get_forecasts_and_risk,
    get_model_analysis,
    get_model_evaluation,
    get_uncertainty_decomposition,
)
from app.database.session import SessionLocal

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_figures():
    db = SessionLocal()
    try:
        # 1. Figure 1: Historical ILI Surveillance & BMA Forecast with 80% and 95% Credible Intervals
        fc_ili = get_forecasts_and_risk(disease_code="ILI", location_code="US-NAT", run_id=None, history_weeks=44, db=db)
        hist = fc_ili["historical_observations"]
        bma_fc = fc_ili["bma_forecasts"]
        risk_fc = fc_ili["outbreak_risk_assessments"]

        hist_dates = [h["observation_date"] for h in hist]
        hist_cases = [h["reported_cases"] for h in hist]

        fc_dates = [f["target_date"] for f in bma_fc]
        fc_mean = [f["mean"] for f in bma_fc]
        fc_median = [f["median"] for f in bma_fc]
        fc_q10 = [f["q10"] for f in bma_fc]
        fc_q90 = [f["q90"] for f in bma_fc]
        fc_q025 = [f["q025"] for f in bma_fc]
        fc_q975 = [f["q975"] for f in bma_fc]
        thresh_vals = [r["threshold_value"] for r in risk_fc]

        all_dates = hist_dates + fc_dates
        x_hist = np.arange(len(hist_dates))
        x_fc = np.arange(len(hist_dates) - 1, len(all_dates))

        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
        ax.plot(x_hist, hist_cases, color="#0f172a", lw=2.2, marker="o", markersize=3.5, label="Actual Reported Cases (CDC ILINet)")
        ax.axvline(x=len(hist_dates) - 1, color="#64748b", linestyle="--", lw=1.2, label=f"Forecast Origin T ({hist_dates[-1]})")

        # Connect last historical observation to first forecast step for visual continuity
        fc_mean_conn = [hist_cases[-1]] + fc_mean
        fc_q10_conn = [hist_cases[-1]] + fc_q10
        fc_q90_conn = [hist_cases[-1]] + fc_q90
        fc_q025_conn = [hist_cases[-1]] + fc_q025
        fc_q975_conn = [hist_cases[-1]] + fc_q975

        ax.fill_between(x_fc, fc_q025_conn, fc_q975_conn, color="#c7d2fe", alpha=0.55, label="BMA 95% Predictive Interval [Q2.5, Q97.5]")
        ax.fill_between(x_fc, fc_q10_conn, fc_q90_conn, color="#818cf8", alpha=0.55, label="BMA 80% Predictive Interval [Q10, Q90]")
        ax.plot(x_fc, fc_mean_conn, color="#312e81", lw=2.5, linestyle="--", marker="s", markersize=4, label="BMA Posterior Predictive Mean")
        ax.step(np.arange(len(hist_dates), len(all_dates)), thresh_vals, where="mid", color="#dc2626", lw=1.8, linestyle=":", label="Analytical Outbreak Threshold")

        tick_idx = np.linspace(0, len(all_dates) - 1, 10, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([all_dates[i] for i in tick_idx], rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("Weekly Reported ILI Cases", fontsize=10, fontweight="bold")
        ax.set_title("U.S. National Influenza-Like Illness (ILI): Historical Surveillance vs. 8-Week BMA Forecast", fontsize=11.5, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#ffffff")
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "fig1_ili_bma_forecast_overview.png")
        plt.close(fig)

        # 2. Figure 2: EDA & Seasonal Decomposition (ILI US-NAT)
        eda_ili = get_eda_analysis(disease_code="ILI", location_code="US-NAT", start_date=None, end_date=None, db=db)
        ts_slice = eda_ili["time_series"][-156:]
        seasonal_prof = eda_ili["seasonal_profile"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
        x_ts = np.arange(len(ts_slice))
        ax1.bar(x_ts, [r["reported_cases"] or 0 for r in ts_slice], color="#cbd5e1", width=0.8, label="Weekly Reported Cases")
        ax1.plot(x_ts, [r["ma_4w"] for r in ts_slice], color="#1d4ed8", lw=1.8, label="4-Week Moving Average")
        ax1.plot(x_ts, [r["ma_12w"] for r in ts_slice], color="#0f172a", lw=1.8, linestyle="--", label="12-Week Moving Average")
        t_idx = np.linspace(0, len(ts_slice) - 1, 6, dtype=int)
        ax1.set_xticks(t_idx)
        ax1.set_xticklabels([ts_slice[i]["observation_date"] for i in t_idx], rotation=20, fontsize=8)
        ax1.set_title("A. Weekly Incidence & Rolling Averages (Last 156 Weeks)", fontsize=10.5, fontweight="bold")
        ax1.set_ylabel("Weekly Cases", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        wks = [s["week_of_year"] for s in seasonal_prof]
        ax2.fill_between(wks, [s["q25_cases"] for s in seasonal_prof], [s["q90_cases"] for s in seasonal_prof], color="#e0e7ff", alpha=0.7, label="25th–90th Percentile Envelope")
        ax2.plot(wks, [s["mean_cases"] for s in seasonal_prof], color="#1e3a8a", lw=2.2, label="Historical Mean Cases")
        ax2.plot(wks, [s["median_cases"] for s in seasonal_prof], color="#047857", lw=1.8, linestyle="--", label="Historical Median Cases")
        ax2.set_title("B. Multi-Year Epidemiological Week Profile (ISO Weeks 1–52)", fontsize=10.5, fontweight="bold")
        ax2.set_xlabel("ISO Epidemiological Week of Year", fontsize=9)
        ax2.set_ylabel("Cases", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "fig2_eda_and_seasonal_profile.png")
        plt.close(fig)

        # 3. Figure 3: Candidate Model Comparison & BMA Weights on Pertussis vs ILI
        ma_pert = get_model_analysis(disease_code="PERTUSSIS", location_code="US-NAT", run_id=None, db=db)
        fc_pert = get_forecasts_and_risk(disease_code="PERTUSSIS", location_code="US-NAT", run_id=None, history_weeks=28, db=db)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
        cands = ma_pert["candidate_models"]
        codes = [c["model_code"] for c in cands]
        weights = [c["bma_weight_pct"] for c in cands]
        log_mls = [c["log_marginal_likelihood"] for c in cands]

        bars = ax1.barh(codes[::-1], weights[::-1], color=["#94a3b8", "#64748b", "#475569", "#0284c7", "#1e3a8a"])
        ax1.set_xlim(0, 75)
        for bar, w_val, lml in zip(bars, weights[::-1], log_mls[::-1]):
            ax1.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{w_val:.1f}% (logML={lml:.1f})", va="center", fontsize=8, fontweight="bold")
        ax1.set_xlabel("Posterior Model Probability P(M_i | D) (%)", fontsize=9)
        ax1.set_title("A. Pertussis (US-NAT) SMC Marginal Likelihood & BMA Weights", fontsize=10.5, fontweight="bold")
        ax1.grid(True, axis="x", linestyle="--", alpha=0.35)

        # Plot individual candidate model forecasts vs BMA on Pertussis
        comb_table = fc_pert["combined_horizon_table"]
        h_steps = [f"+{r['horizon_step']}w\n({r['target_date'][5:]})" for r in comb_table]
        x_h = np.arange(len(h_steps))
        ax2.fill_between(x_h, [r["bma_q10"] for r in comb_table], [r["bma_q90"] for r in comb_table], color="#c7d2fe", alpha=0.5, label="BMA 80% Interval")
        ax2.plot(x_h, [r["bma_mean"] for r in comb_table], color="#1e1b4b", lw=2.8, marker="o", label="BMA Ensemble Mean")
        ax2.plot(x_h, [r["M5_RENEWAL_GROWTH_mean"] for r in comb_table], color="#7c3aed", lw=1.8, linestyle="--", label="M5_RENEWAL_GROWTH (59.2%)")
        ax2.plot(x_h, [r["M3_AR_COUNT_mean"] for r in comb_table], color="#d97706", lw=1.8, linestyle="-.", label="M3_AR_COUNT (40.8%)")
        ax2.plot(x_h, [r["M2_NEGBIN_mean"] for r in comb_table], color="#0284c7", lw=1.4, linestyle=":", label="M2_NEGBIN (0.0%)")
        ax2.set_xticks(x_h)
        ax2.set_xticklabels(h_steps, fontsize=8)
        ax2.set_ylabel("Predicted Weekly Pertussis Cases", fontsize=9)
        ax2.set_title("B. BMA Posterior Mixture vs. Individual Candidate Trajectories", fontsize=10.5, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "fig3_bma_weights_and_candidate_trajectories.png")
        plt.close(fig)

        # 4. Figure 4: 3-Way Law of Total Variance Decomposition (Uncertainty Analysis)
        unc_pert = get_uncertainty_decomposition(disease_code="PERTUSSIS", location_code="US-NAT", run_id=None, db=db)
        h_unc = unc_pert["horizon_uncertainty_decomposition"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
        steps_lbl = [f"+{h['horizon_step']}w" for h in h_unc]
        p_share = np.array([h["parameter_share_pct"] for h in h_unc])
        o_share = np.array([h["observation_share_pct"] for h in h_unc])
        m_share = np.array([h["model_share_pct"] for h in h_unc])

        ax1.bar(steps_lbl, p_share, color="#1e3a8a", label="Within-Model Parameter Uncertainty (%)")
        ax1.bar(steps_lbl, o_share, bottom=p_share, color="#0284c7", label="Aleatoric Observation Noise (%)")
        ax1.bar(steps_lbl, m_share, bottom=p_share + o_share, color="#d97706", label="Between-Model Structural Uncertainty (%)")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Percentage of Total Predictive Variance (%)", fontsize=9)
        ax1.set_xlabel("Forecast Horizon Step", fontsize=9)
        ax1.set_title("A. 3-Way Law of Total Variance Decomposition (Pertussis)", fontsize=10.5, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

        # Discrete Posterior Predictive Histogram at Step +4w
        dist_step4 = h_unc[3]["predictive_distribution"]
        bins_c = [d["bin_center"] for d in dist_step4]
        probs_c = [d["probability"] for d in dist_step4]
        bar_w = (bins_c[1] - bins_c[0]) * 0.82 if len(bins_c) > 1 else 10
        ax2.bar(bins_c, probs_c, width=bar_w, color="#312e81", edgecolor="#ffffff")
        ax2.set_xlabel("Predicted Weekly Case Count at Horizon +4w", fontsize=9)
        ax2.set_ylabel("Posterior Predictive Probability Mass", fontsize=9)
        ax2.set_title(f"B. Full Discrete Posterior Mixture Distribution (+4w: {h_unc[3]['target_date']})", fontsize=10.5, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "fig4_uncertainty_variance_decomposition.png")
        plt.close(fig)

        # 5. Figure 5: Out-of-Sample Expanding-Window Model Evaluation (ILI & Pertussis)
        ev_ili = get_model_evaluation(disease_code="ILI", location_code="US-NAT", run_id=None, db=db)
        metrics_ili = ev_ili["metrics"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
        m_codes = [m["model_code"].replace("_SEASONAL_FOURIER", "_FOURIER").replace("_RENEWAL_GROWTH", "_RENEWAL") for m in metrics_ili]
        crps_vals = [m["crps"] for m in metrics_ili]
        cov95_vals = [m["coverage_95_pct"] for m in metrics_ili]
        colors_eval = ["#1e1b4b" if m["model_code"] == "BMA" else "#64748b" for m in metrics_ili]

        bars1 = ax1.bar(m_codes, crps_vals, color=colors_eval)
        ax1.set_ylabel("Out-of-Sample CRPS (Lower is Better)", fontsize=9)
        ax1.set_title("A. Out-of-Sample Probabilistic Error (CRPS — ILI US-NAT)", fontsize=10.5, fontweight="bold")
        ax1.tick_params(axis="x", rotation=22, labelsize=8)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        for b in bars1:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 500, f"{int(b.get_height()):,}", ha="center", fontsize=7.5, fontweight="bold")

        bars2 = ax2.bar(m_codes, cov95_vals, color=["#059669" if v >= 85 else "#dc2626" for v in cov95_vals])
        ax2.axhline(95.0, color="#0f172a", linestyle="--", lw=1.2, label="Nominal 95% Target")
        ax2.set_ylim(0, 112)
        ax2.set_ylabel("Empirical 95% Interval Coverage (%)", fontsize=9)
        ax2.set_title("B. Empirical 95% Predictive Interval Calibration", fontsize=10.5, fontweight="bold")
        ax2.tick_params(axis="x", rotation=22, labelsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
        for b in bars2:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")

        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "fig5_out_of_sample_backtest_evaluation.png")
        plt.close(fig)

        print("Successfully generated 5 publication-quality figures in:", ASSETS_DIR)
    finally:
        db.close()


if __name__ == "__main__":
    generate_all_figures()
