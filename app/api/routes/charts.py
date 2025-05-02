from datetime import datetime
from typing import Annotated, Optional
from fastapi import APIRouter, Path, Query, HTTPException
from sqlalchemy.orm import selectinload
from sqlmodel import select
from app.api.deps import SessionDep
from app.models import Chart, Stock
from app.schemas import StockRead

router = APIRouter(prefix="/charts", tags=["charts"])


@router.get("/{symbol}", response_model=StockRead)
async def get_stock_with_charts(
        session: SessionDep,
        symbol: Annotated[str, Path(title="Stock symbol, e.g., AAPL")],
        start: Optional[datetime] = Query(None, description="Filter charts from this date (ISO 8601 format)")
):
    stmt = select(Stock).where(Stock.symbol == symbol).options(selectinload(Stock.charts))
    stock = session.execute(stmt).scalar_one_or_none()

    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    if start:
        stock.charts = [chart for chart in stock.charts if chart.date >= start.date()]

    return stock
