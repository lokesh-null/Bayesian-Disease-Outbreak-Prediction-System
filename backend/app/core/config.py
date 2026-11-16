"""Application configuration and environment settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    PROJECT_NAME: str = "Disease Outbreak Prediction System Using Bayesian Model Averaging"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = f"sqlite:///{(DATA_DIR / 'bma_outbreak_system.db').as_posix()}"
    DEFAULT_RANDOM_SEED: int = 42
    SMC_DRAWS_DEFAULT: int = 800
    SMC_CHAINS_DEFAULT: int = 2
    RHAT_WARNING_THRESHOLD: float = 1.05
    RHAT_EXCLUSION_THRESHOLD: float = 1.15
    MIN_ESS_THRESHOLD: float = 50.0
    BMA_POSTERIOR_SAMPLES: int = 2000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

