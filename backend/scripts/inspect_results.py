from app.database.session import SessionLocal
from app.api.routes import get_model_evaluation, get_model_analysis

db = SessionLocal()
for d in ["ILI", "PERTUSSIS", "CAMPYLOBACTER"]:
    ma = get_model_analysis(disease_code=d, location_code="US-NAT", run_id=None, db=db)
    ev = get_model_evaluation(disease_code=d, location_code="US-NAT", run_id=None, db=db)
    print("===", d, "===")
    for m in ma["candidate_models"]:
        print(f"  {m['model_code']}: logML={m['log_marginal_likelihood']} weight={m['bma_weight_pct']}% rhat={m['diagnostics']['max_rhat']}")
    print("  Summary:", ev["comparison_summary_text"])
    for em in ev["metrics"]:
        print(f"  Eval {em['model_code']}: CRPS={em['crps']} NLPD={em['nlpd']} MAE={em['mae']} RMSE={em['rmse']} MASE={em['mase']} Cov80={em['coverage_80_pct']}% Cov95={em['coverage_95_pct']}%")
db.close()

