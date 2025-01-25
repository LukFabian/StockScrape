from datetime import datetime

from pydantic import BaseModel


class Chart(BaseModel):
    high: int
    low: int
    open: int
    close: int
    volume: int
    date: datetime
    symbol: str
    adx_14: float | None
    adx_120: float | None
    rsi_14: float | None
    rsi_120: float | None
