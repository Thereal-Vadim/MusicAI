"""API settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: apps/api/musicai_api/settings.py → ../../../
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    musicai_data_dir: Path = _PROJECT_ROOT / "data"
    musicai_database_url: str = f"sqlite+aiosqlite:///{(_PROJECT_ROOT / 'data' / 'musicai.db').as_posix()}"
    musicai_api_host: str = "0.0.0.0"
    musicai_api_port: int = 8000
    musicai_web_url: str = "http://localhost:3000"
    youtube_ingest_enabled: bool = True

    @property
    def jobs_dir(self) -> Path:
        return self.musicai_data_dir / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.musicai_data_dir / "uploads"


settings = Settings()
settings.musicai_data_dir.mkdir(parents=True, exist_ok=True)
