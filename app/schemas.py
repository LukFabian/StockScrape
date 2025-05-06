from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class ChartRead(BaseModel):
    date: datetime
    symbol: str
    high: int
    low: int
    open: int
    close: int
    volume: int
    adx_14: Optional[float] = None
    adx_120: Optional[float] = None
    dmi_positive_14: Optional[float] = None
    dmi_negative_14: Optional[float] = None
    dmi_positive_120: Optional[float] = None
    dmi_negative_120: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_120: Optional[float] = None

    class Config:
        from_attributes=True


class StockRead(BaseModel):
    symbol: str
    last_modified: datetime = None
    charts: List[ChartRead] = []

    class Config:
        from_attributes=True


class StockPerformanceRead(StockRead):
    performance: float
