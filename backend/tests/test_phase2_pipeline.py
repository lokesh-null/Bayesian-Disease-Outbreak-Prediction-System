"""
Phase 2 Unit & Integration Tests:
Tests Data Ingestion (CSV, Excel, Genuine Datasets), Missing-Value Tracking,
Chronological Preprocessing, Leakage Prevention, Database Persistence, and Exploratory Data Analysis.
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.ml.data_ingestion.ingestor import (
    ingest_file_bytes,
    normalize_surveillance_dataframe,
)
from app.ml.exploratory_analysis.eda import perform_exploratory_analysis
from app.ml.preprocessing.preprocessor import ChronologicalPreprocessor


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


def test_genuine_cdc_ilinet_dataset_ingestion(memory_db):
    """Verify genuine CDC ILINet dataset loads, normalizes, and persists cleanly."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "cdc_ilinet_weekly_2017_2024.csv"
    assert csv_path.exists(), "Genuine CDC ILINet CSV must exist in backend/data/"
    content = csv_path.read_bytes()
    dataset, report = ingest_file_bytes(
        db=memory_db,
        file_bytes=content,
        filename=csv_path.name,
        dataset_name="CDC ILINet Weekly Surveillance (2017-2024)",
        source_name="CDC ILINet via CMU Delphi Epidata API",
    )
    assert dataset.id is not None
    assert dataset.row_count >= 2500
    assert "ILI" in report["diseases_detected"]
    assert "US-NAT" in report["locations_detected"]


def test_excel_ingestion_and_duplicate_and_missing_handling(memory_db):
    """Verify Excel (.xlsx) ingestion, duplicate removal, and explicit missing week gap detection."""
    raw = pd.DataFrame({
        "date": [
            "2023-01-02",
            "2023-01-09",
            "2023-01-09",  # Duplicate date
            "2023-01-16",
            # 2023-01-23 is omitted (missing week gap!)
            "2023-01-30",
            "2023-02-06",
            "2023-02-13",
            "2023-02-20",
            "2023-02-27",
        ],
        "disease": ["Dengue"] * 9,
        "region": ["US-PR"] * 9,
        "cases": [12, 15, 18, None, 22, 19, 25, 30, 28],
    })
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    xlsx_bytes = buf.getvalue()

    ds, report = ingest_file_bytes(
        db=memory_db,
        file_bytes=xlsx_bytes,
        filename="test_surveillance.xlsx",
    )
    assert report["duplicate_count"] == 1
    assert report["explicit_gap_rows_inserted"] == 1
    # 1 explicit NA in input + 1 inserted missing week gap = 2 missing observations tracked
    assert report["missing_case_count"] == 2
    assert ds.missing_count == 2


def test_negative_counts_rejected():
    """Rule 7/Mathematical verification: negative case counts must be rejected."""
    bad_df = pd.DataFrame({
        "observation_date": ["2023-01-02", "2023-01-09"],
        "confirmed_cases": [10, -5],
    })
    with pytest.raises(ValueError, match="Negative case counts"):
        normalize_surveillance_dataframe(bad_df)


def test_chronological_preprocessor_no_future_leakage():
    """Verify preprocessor learns scaling and lags strictly from training data without leaking test data."""
    dates = pd.date_range("2022-01-03", periods=30, freq="7D").strftime("%Y-%m-%d")
    # Training has small values (around 20), test has massive spike (10,000)
    cases = [20 + (i % 5) for i in range(20)] + [10000 for _ in range(10)]
    df = pd.DataFrame({"observation_date": dates, "confirmed_cases": cases, "epiweek": [202201 + i for i in range(30)]})

    train_df, test_df = ChronologicalPreprocessor.chronological_split(df, 20)
    assert len(train_df) == 20
    assert len(test_df) == 10
    assert train_df["observation_date"].max() < test_df["observation_date"].min()

    prep = ChronologicalPreprocessor()
    feats, state = prep.fit_transform_train(train_df)
    assert state.y_train_mean < 25.0  # Must NOT be influenced by 10,000 in test_df
    assert state.last_observed_counts[-1] == cases[19]


def test_eda_module_on_genuine_data():
    """Verify EDA generates all required statistical summaries, seasonal profiles, and anomaly lists."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "cdc_ilinet_weekly_2017_2024.csv"
    df = pd.read_csv(csv_path)
    nat_df = df[df["location_code"] == "US-NAT"].copy()
    eda_res = perform_exploratory_analysis(nat_df, all_locations_df=df)

    assert eda_res["summary_statistics"]["valid_observations"] > 400
    assert eda_res["summary_statistics"]["overdispersed"] is True
    assert eda_res["summary_statistics"]["variance_to_mean_ratio"] > 10.0
    assert len(eda_res["monthly_trends"]) > 50
    assert len(eda_res["seasonal_profile"]) >= 52
    assert len(eda_res["geographic_comparison"]) == 6

