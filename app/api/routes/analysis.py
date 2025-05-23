from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Query, Path, HTTPException

from app.api.analyze.linear_regression import analyze_linear_regression
from app.api.analyze.sgp_lstm import analyze_sgp_lstm
from app.api.deps import SessionDep
from app.api.routes.stock import get_stock
from app.schemas import StockRead

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.put("/stock/{symbol}")
async def put_stock_analysis(session: SessionDep, symbol: Annotated[str, Path(title="The unique symbol of the stock")],
                             mode: str = Query(..., description="Choose 'LINEAR REGRESSION' or 'SGP LSTM'"),
                             start: Optional[datetime] = Query(None,
                                                               description="Filter charts from this date (ISO 8601 format)"),
                             days_to_predict: Optional[int] = 14) -> StockRead:
    match mode:
        case "LINEAR REGRESSION":
            return analyze_linear_regression(await get_stock(session, symbol, with_technicals=False), start,
                                             days_to_predict)
        case "SGP LSTM":
            return await analyze_sgp_lstm(await get_stock(session, symbol, with_technicals=True), start, session,
                                          days_to_predict)
        case _:
            raise HTTPException(status_code=400, detail=f"specified mode unknown: {mode}")
