from typing import Annotated, Optional

import pytz
import requests
from fastapi import APIRouter, Path, Query, HTTPException
from sqlalchemy import func
from app.api.deps import SessionDep
from app.core.config import settings
from app.models import Stock, Chart
from app.schemas import StockRead
from app.utils import calculate_technical_stock_data, process_stocks_from_alphavantage

router = APIRouter(prefix="/stock", tags=["stock"])


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
    for chart in stock_read.charts:
        chart.date = chart.date.replace(tzinfo=pytz.UTC)
    if with_technicals:
        stock_read = calculate_technical_stock_data(stock_read)
    return stock_read

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