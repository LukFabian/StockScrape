from typing import Annotated

from fastapi import APIRouter, Path, HTTPException
from sqlalchemy import func

from app.api.deps import SessionDep
from app.models import Stock, Chart

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/{symbol}", response_model=Stock)
async def get_stock(session: SessionDep,
                    symbol: Annotated[str, Path(title="The unique symbol of the stock to retrieve")]):
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
