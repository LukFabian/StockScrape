from fastapi import APIRouter
from app.api.routes import stocks, charts

api_router = APIRouter()
api_router.include_router(router=stocks.router)
api_router.include_router(router=charts.router)
