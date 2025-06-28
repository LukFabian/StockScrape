import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pyinstrument import Profiler
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.api.deps import db_manager
from app.api.main import api_router
from app.api.routes.stock import put_stock
from app.core.config import settings
from app.logger import logger
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
            try:
                await put_stock(session, symbol, time_frame="DAILY")
            except ValueError as e:
                logger.warning(e)
                break

    yield

class PyInstrumentMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        profiler = Profiler(interval=0.001, async_mode="enabled")
        profiler.start()
        try:
            response = await call_next(request)
            profiler.stop()
            # Write result to html file
            profiler.write_html("profile.html")
            return response
        finally:
            profiler.stop()
            # Write result to html file
            profiler.write_html("profile.html")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/api/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan
)

#app.add_middleware(PyInstrumentMiddleWare)

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
