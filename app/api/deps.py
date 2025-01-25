from collections.abc import Generator
from sqlmodel import Session
from app.core.config import settings
from database.manager import DatabaseManager
from fastapi import Depends
from typing import Annotated

db_manager = DatabaseManager(settings.DB_URL, settings.ALEMBIC_PATH)


def get_db() -> Generator[Session, None, None]:
    with db_manager.get_session() as session:
        yield session

SessionDep = Annotated[Session, Depends(get_db)]
