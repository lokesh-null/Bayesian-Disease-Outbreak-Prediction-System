"""
Phase 6 & 8 End-to-End API Integration Tests:
Tests the complete FastAPI application workflow:
Dataset retrieval -> Custom CSV upload -> Disease/Location listing -> EDA endpoint ->
Model analysis & BMA posterior weights -> Outbreak forecast & risk classification ->
3-Way Uncertainty decomposition -> Chronological evaluation metrics -> Overview dashboard.
"""
import io
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_health_and_datasets_endpoints():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

        ds_resp = client.get("/api/v1/datasets")
        assert ds_resp.status_code == 200
        datasets = ds_resp.json()
        assert len(datasets) >= 2
        names = [d["name"] for d in datasets]
        assert any("ILINet" in n for n in names)
        assert any("NNDSS" in n for n in names)


def test_diseases_locations_observations_and_eda_endpoints():
    with TestClient(app) as client:
        d_resp = client.get("/api/v1/diseases")
        assert d_resp.status_code == 200
        diseases = d_resp.json()
        codes = [d["code"] for d in diseases]
        assert "ILI" in codes
        assert "PERTUSSIS" in codes

        l_resp = client.get("/api/v1/locations?disease_code=ILI")
        assert l_resp.status_code == 200
        locs = l_resp.json()
        assert any(l["code"] == "US-NAT" for l in locs)

        obs_resp = client.get("/api/v1/observations?disease_code=ILI&location_code=US-NAT&limit=50")
        assert obs_resp.status_code == 200
        obs_data = obs_resp.json()
        assert obs_data["returned_observations"] == 50

        eda_resp = client.get("/api/v1/eda?disease_code=ILI&location_code=US-NAT")
        assert eda_resp.status_code == 200
        eda_data = eda_resp.json()
        assert eda_data["summary_statistics"]["valid_observations"] > 400
        assert len(eda_data["seasonal_profile"]) >= 52


def test_model_analysis_forecasts_uncertainty_and_evaluation_endpoints():
    with TestClient(app) as client:
        ma_resp = client.get("/api/v1/models/analysis?disease_code=ILI&location_code=US-NAT")
        assert ma_resp.status_code == 200
        ma_data = ma_resp.json()
        assert len(ma_data["candidate_models"]) == 5
        assert abs(ma_data["bma_weight_sum"] - 1.0) < 1e-4

        fc_resp = client.get("/api/v1/forecasts?disease_code=ILI&location_code=US-NAT")
        assert fc_resp.status_code == 200
        fc_data = fc_resp.json()
        assert len(fc_data["bma_forecasts"]) == 8
        assert len(fc_data["outbreak_risk_assessments"]) == 8

        unc_resp = client.get("/api/v1/uncertainty?disease_code=ILI&location_code=US-NAT")
        assert unc_resp.status_code == 200
        unc_data = unc_resp.json()
        assert len(unc_data["horizon_uncertainty_decomposition"]) == 8
        for step in unc_data["horizon_uncertainty_decomposition"]:
            total_pct = step["parameter_share_pct"] + step["observation_share_pct"] + step["model_share_pct"]
            assert 99.5 <= total_pct <= 100.5

        ev_resp = client.get("/api/v1/evaluation?disease_code=ILI&location_code=US-NAT")
        assert ev_resp.status_code == 200
        ev_data = ev_resp.json()
        assert ev_data["has_validation"] is True
        assert len(ev_data["metrics"]) == 6

        ov_resp = client.get("/api/v1/overview?disease_code=ILI&location_code=US-NAT")
        assert ov_resp.status_code == 200
        ov_data = ov_resp.json()
        assert ov_data["disease"]["code"] == "ILI"
        assert ov_data["data_freshness"]["latest_reported_cases"] > 0


def test_custom_dataset_upload_and_sync_training_workflow():
    """Full integration test: upload custom surveillance CSV -> trigger synchronous training -> verify BMA forecast."""
    with TestClient(app) as client:
        dates = pd.date_range("2023-01-02", periods=36, freq="7D").strftime("%Y-%m-%d")
        cases = [18 + int(8 * ((i % 12) / 12.0)) + (i % 4) for i in range(36)]
        df = pd.DataFrame({
            "observation_date": dates,
            "disease_code": ["MEASLES_TEST"] * 36,
            "disease_name": ["Measles Surveillance Validation Series"] * 36,
            "location_code": ["US-TEST"] * 36,
            "location_name": ["Test Jurisdiction"] * 36,
            "confirmed_cases": cases,
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        up_resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("measles_test.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"dataset_name": "Measles Test Series", "source_name": "Integration Test Suite"},
        )
        assert up_resp.status_code == 200

        tr_resp = client.post(
            "/api/v1/training/run?sync=true",
            json={
                "disease_code": "MEASLES_TEST",
                "location_code": "US-TEST",
                "forecast_horizon": 4,
                "smc_draws": 150,
                "smc_chains": 2,
                "random_seed": 42,
                "run_backtest": False,
            },
        )
        assert tr_resp.status_code == 200
        tr_data = tr_resp.json()
        assert tr_data["status"] == "COMPLETED"

        fc_resp = client.get("/api/v1/forecasts?disease_code=MEASLES_TEST&location_code=US-TEST")
        assert fc_resp.status_code == 200
        assert len(fc_resp.json()["bma_forecasts"]) == 4

