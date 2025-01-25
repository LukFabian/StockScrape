from fastapi import APIRouter
from app.api.routes import stocks

api_router = APIRouter()
api_router.include_router(router=stocks.router)
