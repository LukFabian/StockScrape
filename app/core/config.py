import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

file_path = pathlib.Path(__file__).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use top level .env file
        env_file=file_path.parent.parent.parent.joinpath(".env"),
        env_ignore_empty=True,
        extra="ignore",
    )
    DB_PORT: int = 0
    DB_HOST: str = ""
    DB_URL: str = ""
    ALEMBIC_PATH: pathlib.PosixPath = pathlib.Path(file_path.parent.parent.resolve(), "alembic.ini").resolve()
    PROJECT_NAME: str = "StockScrape"
    ALPHAVANTAGE_API_KEY: str = ""


settings = Settings()  # type: ignore
