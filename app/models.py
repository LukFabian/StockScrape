from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, String, Float, Boolean, DateTime, Integer


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stock"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    last_modified: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    charts: Mapped[List["Chart"]] = relationship(back_populates="stock", cascade="all, delete-orphan", order_by="Chart.date")


class Chart(Base):
    __tablename__ = "chart"

    date: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stock.symbol"), primary_key=True)

    high: Mapped[int] = mapped_column(Integer, nullable=False)
    low: Mapped[int] = mapped_column(Integer, nullable=False)
    open: Mapped[int] = mapped_column(Integer, nullable=False)
    close: Mapped[int] = mapped_column(Integer, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)

    adx_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adx_120: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi_120: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    stock: Mapped["Stock"] = relationship(back_populates="charts")