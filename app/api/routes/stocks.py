import datetime
from typing import Annotated, Optional
import json
import requests
from fastapi import APIRouter, Path, HTTPException, Query
from sqlmodel import select
from sqlalchemy.orm import aliased
from sqlalchemy import func, desc, asc
from app.api.deps import SessionDep
from app.core.config import settings
from app.models import Stock, Chart
from app.schemas import StockRead
from app.utils import process_stocks_from_alphavantage

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}", response_model=StockRead)
async def get_stock(session: SessionDep,
                    symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve from database")]):
    stock = (
        session.query(Stock)
        .join(Chart, Stock.symbol == Chart.symbol)  # Join `Stock` with `Chart`
        .filter(Stock.symbol == symbol)  # Filter by the given symbol
        .group_by(Stock.symbol)  # Group by `Stock.symbol`
        .order_by(func.max(Chart.date).desc())
        .one_or_none()  # Fetch one row or return `None`
    )

    if not stock:
        raise HTTPException(status_code=404, detail=f"stock not found with symbol: {symbol}")
    return stock


@router.get("/", response_model=StockRead)
async def get_stocks(session: SessionDep, mode: str = Query(..., description="Choose between 'best' or 'worst'"),
                     start: Optional[datetime.datetime] = Query(None,
                                                                description="ISO 8601, e.g. 2025-04-24T09:30:00Z")):
    # 1. Validate input
    start = start if start else datetime.datetime.now()  - datetime.timedelta(days=7)
    start = start.date()
    if mode not in {"best", "worst"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'best' or 'worst'.")

    # Aliases for self-join
    chart_start = aliased(Chart)
    chart_latest = aliased(Chart)

    # 2. Subquery: calculate performance for each stock within timeframe
    subquery = (
        select(
            chart_latest.symbol.label("symbol"),
            ((chart_latest.close - chart_start.close) / chart_start.close).label("performance")
        )
        .join(
            chart_start,
            (chart_latest.symbol == chart_start.symbol)
            & (chart_start.date == start)
        )
        .where(
            chart_latest.date == select(func.max(Chart.date))
                                .where(Chart.symbol == chart_latest.symbol)
                                .scalar_subquery()
        )
    ).subquery()

    # 3. Select best or worst performer
    order_clause = desc(subquery.c.performance) if mode == "best" else asc(subquery.c.performance)

    stmt = select(subquery).order_by(order_clause).limit(1)

    symbol = session.execute(stmt).scalar_one_or_none()
    if not symbol:
        raise HTTPException(status_code=404, detail="No stocks found for the given timeframe.")

    return await get_stock(session, symbol)


@router.get("/count/all", response_model=int)
async def get_stocks_count(session: SessionDep):
    stmt = select(func.count(Stock.symbol))
    return session.execute(stmt).scalar_one_or_none()


@router.put("/{symbol}", response_model=StockRead)
async def put_stock(session: SessionDep,
                    symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve from alphavantage")],
                    time_frame: str = Query(..., description="Choose between 'MONTHLY' or 'WEEKLY' or 'DAILY'")):
    api_key = settings.ALPHAVANTAGE_API_KEY
    match time_frame:
        case "MONTHLY":
            request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={api_key}"
        case "WEEKLY":
            request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={api_key}"
        case "DAILY":
            request_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
        case _:
            raise HTTPException(status_code=400, detail=f"wrong time_frame specified: {time_frame}")

    data = requests.get(request_url).json()
    with open(f'data_{time_frame}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    process_stocks_from_alphavantage(session, data, time_frame)
    return await get_stock(session, symbol)
