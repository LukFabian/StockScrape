from typing import Annotated

from fastapi import APIRouter, Path
from sqlalchemy import func

from app.api.deps import SessionDep
from app.models.stock import Stock as AppStock
from database.models import Stock, Chart

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}", response_model=AppStock)
async def get_stock(session: SessionDep,
                    symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve")]):
    stock = (
        session.query(Stock)
        .join(Chart, Stock.symbol == Chart.symbol)  # Join with the related `charts` table
        .group_by(Stock.symbol, Chart.date)  # Group by `Stock` to count related rows
        .having(Stock.symbol == symbol)  # Only include stocks with charts
        .order_by(Chart.date)
        .one()
    )
    # convert to valid pydantic object
    """
    stock = AppStock(
        symbol=symbol,
        name=stock.name,
        industry=stock.industry,
        marketcap=stock.marketcap,
        isNasdaq100=stock.isNasdaq100,
        delta_indicator=stock.deltaIndicator,
        charts=stock.charts
    )
    """
    return stock
