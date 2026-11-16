"""
Seed the SQLite database with the two genuine CDC surveillance datasets (CDC ILINet & CDC NNDSS)
and execute complete, genuine PyMC SMC training + BMA + Outbreak Risk + Expanding-Window Cross-Validation
runs for:
1. Influenza-Like Illness (`ILI`) in United States National (`US-NAT`)
2. Pertussis (`PERTUSSIS`) in United States National (`US-NAT`)
3. Dengue (`DENGUE`) in Puerto Rico (`US-PR`) / United States National (`US-NAT`)
"""
import json
from app.database.session import Base, SessionLocal, engine
from app.models.entities import Disease, Location, TrainingRun
from app.services.pipeline_service import (
    ensure_initial_datasets_and_baseline_runs,
    execute_training_run_pipeline,
)


def seed_all():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        print("1. Ingesting genuine CDC ILINet and CDC NNDSS datasets...")
        ensure_initial_datasets_and_baseline_runs(db)

        targets = [
            ("ILI", "US-NAT", 8, 350, 2, 2, 4),
            ("PERTUSSIS", "US-NAT", 8, 300, 2, 2, 4),
            ("CAMPYLOBACTER", "US-NAT", 8, 300, 2, 2, 4),
        ]

        for d_code, l_code, horizon, draws, chains, bt_folds, bt_horizon in targets:
            disease = db.query(Disease).filter(Disease.code == d_code).first()
            location = db.query(Location).filter(Location.code == l_code).first()
            if not disease or not location:
                print(f"Skipping ({d_code}, {l_code}) - not found")
                continue

            existing = (
                db.query(TrainingRun)
                .filter(
                    TrainingRun.disease_id == disease.id,
                    TrainingRun.location_id == location.id,
                    TrainingRun.status == "COMPLETED",
                )
                .first()
            )
            if existing:
                print(f"Completed run already exists for ({d_code}, {l_code}): run_id={existing.id}")
                continue

            tr = TrainingRun(
                disease_id=disease.id,
                location_id=location.id,
                train_start_date="ALL",
                train_end_date="ALL",
                train_observations_count=150,
                forecast_horizon=horizon,
                random_seed=42,
                sampling_config_json=json.dumps({
                    "smc_draws": draws,
                    "smc_chains": chains,
                    "run_backtest": True,
                    "backtest_folds": bt_folds,
                    "backtest_horizon": bt_horizon,
                    "outbreak_z_multiplier": 1.28,
                }),
                status="PENDING",
            )
            db.add(tr)
            db.commit()
            db.refresh(tr)

            print(f"Running genuine PyMC SMC + BMA + Backtest pipeline for ({d_code}, {l_code}) [run_id={tr.id}]...")
            execute_training_run_pipeline(
                training_run_id=tr.id,
                run_backtest=True,
                backtest_folds=bt_folds,
                backtest_horizon=bt_horizon,
                outbreak_z_multiplier=1.28,
            )
            print(f"Successfully completed run_id={tr.id} for ({d_code}, {l_code})!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_all()

