import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

file_path = pathlib.Path(__file__).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use top level .env file (one level above ./backend/)
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    DB_PORT: int = 0
    DB_HOST: str = ""
    DB_URL: str = ""
    ALEMBIC_PATH: pathlib.PosixPath = pathlib.Path(file_path.parent.parent.resolve(), "alembic.ini").resolve()
    PROJECT_NAME: str = "StockScrape"


settings = Settings()  # type: ignore
