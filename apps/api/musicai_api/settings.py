"""API settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    musicai_data_dir: Path = Path("./data")
    musicai_database_url: str = "sqlite+aiosqlite:///./data/musicai.db"
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
