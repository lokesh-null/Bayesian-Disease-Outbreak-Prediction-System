"""
FastAPI Application Entrypoint for the Disease Outbreak Prediction System Using Bayesian Model Averaging.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.database.session import Base, SessionLocal, engine
from app.services.pipeline_service import ensure_initial_datasets_and_baseline_runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        ensure_initial_datasets_and_baseline_runs(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research-grade epidemiological forecasting and outbreak risk detection platform "
        "powered by PyMC Sequential Monte Carlo (SMC) and Bayesian Model Averaging (BMA)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.VERSION}

