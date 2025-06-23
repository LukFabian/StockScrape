import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from sqlalchemy import select
from app.api.deps import db_manager
from app.api.main import api_router
from app.api.routes.stock import put_stock
from app.core.config import settings
from app.models import Stock


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db_manager.get_session() as session:
        yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
        stmt = (
            select(Stock.symbol)
            .where(Stock.last_modified <= yesterday)
        )
        result = session.execute(stmt).scalars().all()
        for symbol in result:
            await put_stock(session, symbol, time_frame="DAILY")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/api/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan
)

origins = [
    "http://localhost:3000",  # Vite dev server
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(api_router)
