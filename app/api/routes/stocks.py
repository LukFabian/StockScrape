import datetime
from typing import Annotated, Optional
import requests
from fastapi import APIRouter, Path, HTTPException, Query
from sqlmodel import select
from sqlalchemy import func
from app.api.deps import SessionDep
from app.core.config import settings
from app.models import Stock, Chart
from app.schemas import StockRead, StockPerformanceRead
from app.utils import process_stocks_from_alphavantage, calculate_technical_stock_data, get_stock_performance

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}")
async def get_stock(session: SessionDep,
                    symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve from database")],
                    with_technicals: Optional[bool] = Query(False, description="Also calculate the technical stock data")) -> StockRead:
    stock: Stock = (
        session.query(Stock)
        .join(Chart, Stock.symbol == Chart.symbol)  # Join `Stock` with `Chart`
        .filter(Stock.symbol == symbol)  # Filter by the given symbol
        .group_by(Stock.symbol)  # Group by `Stock.symbol`
        .order_by(func.max(Chart.date).desc())
        .one_or_none()  # Fetch one row or return `None`
    )

    if not stock:
        raise HTTPException(status_code=404, detail=f"stock not found with symbol: {symbol}")
    stock_read: StockRead = StockRead.model_validate(stock, from_attributes=True)
    if with_technicals:
        stock_read = calculate_technical_stock_data(stock_read)
    return stock_read

@router.get("/")
async def get_stocks(session: SessionDep, mode: str = Query(..., description="Choose between 'best' or 'worst'"),
                     start: Optional[datetime.datetime] = Query(None,
                                                                description="ISO 8601, e.g. 2025-04-24T09:30:00Z")) -> StockPerformanceRead:
    start = start if start else datetime.datetime.now()  - datetime.timedelta(days=7)
    if mode not in {"best", "worst"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'best' or 'worst'.")

    symbols = session.execute(select(Stock.symbol)).scalars().all()
    result_stock = None
    for symbol in symbols:
        result = get_stock_performance(session=session, stock_symbol=symbol, start_time=start, is_best=mode == "best")
        if not result_stock:
            result_stock = result
        elif result.performance < result_stock.performance and mode == "worst":
            result_stock = result
        elif result.performance > result_stock.performance and mode == "best":
            result_stock = result
    if not result_stock:
        raise HTTPException(status_code=404, detail="No stocks found for the given timeframe.")
    performance = result_stock.performance
    result_stock_with_technical_data = calculate_technical_stock_data(result_stock)
    return StockPerformanceRead(
        symbol=result_stock.symbol,
        performance=performance,
        charts=result_stock_with_technical_data.charts,
        last_modified=result_stock_with_technical_data.last_modified,
    )


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
    process_stocks_from_alphavantage(session, data, time_frame)
    return await get_stock(session, symbol)
