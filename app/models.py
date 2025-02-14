import datetime
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship


class Stock(SQLModel, table=True):
    __tablename__ = "stock"
    symbol: str = Field(primary_key=True, nullable=False)
    name: str = Field(nullable=False)
    industry: Optional[str] = Field(default=None)
    marketcap: Optional[float] = Field(default=None)
    isNasdaq100: Optional[bool] = Field(default=None)
    deltaIndicator: Optional[str] = Field(default=None)

    charts: List["Chart"] = Relationship(back_populates="stock")


class Chart(SQLModel, table=True):
    __tablename__ = "chart"
    high: int = Field(nullable=False)
    low: int = Field(nullable=False)
    open: int = Field(nullable=False)
    close: int = Field(nullable=False)
    volume: int = Field(nullable=False)
    date: datetime.datetime = Field(primary_key=True)
    adx_14: Optional[float] = Field(default=None)
    adx_120: Optional[float] = Field(default=None)
    rsi_14: Optional[float] = Field(default=None)
    rsi_120: Optional[float] = Field(default=None)
    symbol: str = Field(foreign_key="stock.symbol", primary_key=True)

    stock: Stock = Relationship(back_populates="charts")
