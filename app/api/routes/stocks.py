import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.util import symbol
from sqlmodel import select
from sqlalchemy import func
from app.api.deps import SessionDep
from app.models import Stock
from app.schemas import StockPerformanceRead
from app.utils import calculate_technical_stock_data, \
    get_stocks_performance

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/all")
async def get_all_stocks(session: SessionDep, start: Optional[datetime.datetime] = Query(None,
                                                                                         description="ISO 8601, e.g. 2025-04-24T09:30:00Z")) -> \
List[StockPerformanceRead]:
    start = start if start else datetime.datetime.now() - datetime.timedelta(days=7)
    symbols = session.execute(select(Stock.symbol)).scalars().all()
    results = get_stocks_performance(session, stock_symbols=symbols, start_time=start, is_best=True)
    results_with_technical_data = [calculate_technical_stock_data(result) for result in results]
    return results_with_technical_data


@router.get("/")
async def get_stocks(session: SessionDep, mode: str = Query(..., description="Choose between 'best' or 'worst'"),
                     start: Optional[datetime.datetime] = Query(None,
                                                                description="ISO 8601, e.g. 2025-04-24T09:30:00Z")) -> StockPerformanceRead:
    start = start if start else datetime.datetime.now() - datetime.timedelta(days=7)
    if mode not in {"best", "worst"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'best' or 'worst'.")

    symbols = session.execute(select(Stock.symbol)).scalars().all()
    results = get_stocks_performance(session=session, stock_symbols=symbols, start_time=start, is_best=mode == "best")
    if len(results) == 0:
        raise HTTPException(status_code=404, detail=f"stock not found with symbol: {symbol}")
    result = results[0]
    result_stock_with_technical_data = calculate_technical_stock_data(result)
    return result_stock_with_technical_data


@router.get("/count", response_model=int)
async def get_stocks_count(session: SessionDep):
    stmt = select(func.count(Stock.symbol))
    return session.execute(stmt).scalar_one_or_none()
