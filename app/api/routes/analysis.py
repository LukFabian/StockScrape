from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Query, Path, HTTPException

from app.api.analyze.linear_regression import analyze_linear_regression
from app.api.analyze.sgp_lstm import analyze_sgp_lstm
from app.api.deps import SessionDep, db_manager
from app.api.routes.stock import get_stock
from app.schemas import StockRead, ClassificationPrediction

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.put("/stock/regression/{symbol}")
async def put_stock_analysis_regression(session: SessionDep,
                                        symbol: Annotated[str, Path(title="The unique symbol of the stock")],
                                        mode: str = Query(..., description="Choose 'LINEAR REGRESSION'"),
                                        start: Optional[datetime] = Query(None,
                                                                          description="Filter charts from this date (ISO 8601 format)"),
                                        days_to_predict: Optional[int] = 14) -> StockRead:
    match mode:
        case "LINEAR REGRESSION":
            return analyze_linear_regression(await get_stock(session, symbol, with_technicals=False), start,
                                             days_to_predict)
        case _:
            raise HTTPException(status_code=400, detail=f"specified mode unknown: {mode}")


@router.put("stock/classification/{symbol}")
async def put_stock_analysis_classification(session: SessionDep,
                                            symbol: Annotated[str, Path(title="The unique symbol of the stock")],
                                            mode: str = Query(...,
                                                              description="Choose 'SGP LSTM'")) -> ClassificationPrediction:
    match mode:
        case "SGP LSTM":
            return await analyze_sgp_lstm(await get_stock(session, symbol, with_technicals=True), session,
                                          db_manager.SessionLocal)
        case _:
            raise HTTPException(status_code=400, detail=f"specified mode unknown: {mode}")
