from fastapi import APIRouter
from app.api.routes import stocks, charts, stock, analysis

api_router = APIRouter()
api_router.include_router(router=stocks.router)
api_router.include_router(router=stock.router)
api_router.include_router(router=charts.router)
api_router.include_router(router=analysis.router)
