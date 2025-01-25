from pydantic import BaseModel

from app.models.chart import Chart


class Stock(BaseModel):
    symbol: str
    name: str
    industry: str | None
    marketcap: int
    is_nasdaq_100: bool | None
    delta_indicator: str | None
    charts: list[Chart] | None
