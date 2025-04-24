import datetime
from typing import Annotated

import requests
import sqlalchemy
from fastapi import APIRouter, Path, HTTPException, Query
from sqlmodel import select
from sqlalchemy.orm import aliased
from sqlalchemy import func, desc, asc
from app.api.deps import SessionDep
from app.models import Stock, Chart
from app.utils import process_stocks_from_nasdaq

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}", response_model=Stock)
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


@router.put("/{symbol}", response_model=Stock)
async def update_stock(session: SessionDep,
                       symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve from nasdaq")]):
    request_headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0"}
    today = datetime.datetime.now()
    time_delta = datetime.timedelta(days=365 * 5)
    five_years_ago = today - time_delta
    formatting = "%Y-%m-%d"
    five_years_ago_str = five_years_ago.strftime(formatting)
    today_str = today.strftime(formatting)
    request_url = f"https://api.nasdaq.com/api/quote/{symbol.lower()}/chart?assetclass=stocks&fromdate={five_years_ago_str}&todate={today_str}"
    json_resp = requests.get(request_url, headers=request_headers).json()
    if not json_resp:
        raise HTTPException(status_code=404, detail=f"stock not found with symbol: {symbol}")

    stock = process_stocks_from_nasdaq(json_resp.get("data"))
    session.add(stock)
    try:
        session.commit()
    except sqlalchemy.exc.IntegrityError:
        session.rollback()
        existing_stock = session.get(Stock, stock.symbol)
        if existing_stock:
            for key, value in stock.model_dump().items():
                setattr(existing_stock, key, value)
            session.add(existing_stock)
            session.commit()
            return existing_stock
    return stock


@router.get("/", response_model=Stock)
async def get_stocks(session: SessionDep, mode: str = Query(..., description="Choose between 'best' or 'worst'")):
    # Alias for self-join
    chart_prev = aliased(Chart)

    # Build the subquery with performance calculation
    subquery = (
        select(
            Chart.symbol.label("symbol"),
            ((Chart.close - chart_prev.close) / chart_prev.close).label("performance")
        )
        .join(
            chart_prev,
            (Chart.symbol == chart_prev.symbol) &
            (func.date(Chart.date) == func.date(chart_prev.date + datetime.timedelta(days=1)))
        )
    ).subquery()


    # Choose best or worst performer
    if mode == "best":
        stmt = (
            select(subquery)
            .order_by(desc(subquery.c.performance))
            .limit(1)
        )
    elif mode == "worst":
        stmt = (
            select(subquery)
            .order_by(asc(subquery.c.performance))
            .limit(1)
        )
    else:
        return {"error": "Invalid mode. Choose 'best' or 'worst'."}
    stock = session.execute(stmt).first()
    existing_stock = session.get(Stock, stock.symbol)
    if not existing_stock:
        raise HTTPException(status_code=404, detail=f"no stocks exist in database")
    return existing_stock
