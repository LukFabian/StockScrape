from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, String, DateTime, Integer, Float


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stock"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    last_modified: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    charts: Mapped[List["Chart"]] = relationship(back_populates="stock", cascade="all, delete-orphan",
                                                 order_by="Chart.date")
    profile: Mapped[Optional["StockProfile"]] = relationship(
        back_populates="stock",
        uselist=False,
        cascade="all, delete-orphan"
    )

    metrics: Mapped[List["StockMetric"]] = relationship(
        back_populates="stock",
        cascade="all, delete-orphan",
        order_by="StockMetric.date"
    )


class Chart(Base):
    __tablename__ = "chart"

    date: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stock.symbol"), primary_key=True)

    high: Mapped[int] = mapped_column(Integer, nullable=False)
    low: Mapped[int] = mapped_column(Integer, nullable=False)
    open: Mapped[int] = mapped_column(Integer, nullable=False)
    close: Mapped[int] = mapped_column(Integer, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    adx_14: Mapped[Optional[float]] = mapped_column(Float, nullable=False)
    adx_120: Mapped[Optional[float]] = mapped_column(Float, nullable=False)
    dmi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=False)
    dmi_120: Mapped[Optional[float]] = mapped_column(Float, nullable=False)
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=False)
    rsi_120: Mapped[Optional[float]] = mapped_column(Float, nullable=False)

    stock: Mapped["Stock"] = relationship(back_populates="charts")


class StockProfile(Base):
    __tablename__ = "stock_profile"

    symbol: Mapped[str] = mapped_column(ForeignKey("stock.symbol"), primary_key=True)

    asset_type: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    cik: Mapped[str] = mapped_column(String)
    exchange: Mapped[str] = mapped_column(String)
    currency: Mapped[str] = mapped_column(String)
    country: Mapped[str] = mapped_column(String)
    sector: Mapped[str] = mapped_column(String)
    industry: Mapped[str] = mapped_column(String)
    address: Mapped[str] = mapped_column(String)
    official_site: Mapped[str] = mapped_column(String)
    fiscal_year_end: Mapped[str] = mapped_column(String)

    stock: Mapped["Stock"] = relationship(back_populates="profile")


class StockMetric(Base):
    __tablename__ = "stock_metric"

    symbol: Mapped[str] = mapped_column(ForeignKey("stock.symbol"), primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    latest_quarter: Mapped[Optional[DateTime]] = mapped_column(DateTime)
    market_capitalization: Mapped[Optional[int]] = mapped_column(Integer)
    ebitda: Mapped[Optional[int]] = mapped_column(Integer)
    pe_ratio: Mapped[Optional[float]] = mapped_column(Integer)
    peg_ratio: Mapped[Optional[float]] = mapped_column(Integer)
    book_value: Mapped[Optional[float]] = mapped_column(Integer)
    dividend_per_share: Mapped[Optional[float]] = mapped_column(Integer)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Integer)
    eps: Mapped[Optional[float]] = mapped_column(Integer)
    revenue_per_share_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    profit_margin: Mapped[Optional[float]] = mapped_column(Integer)
    operating_margin_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    return_on_assets_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    return_on_equity_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    revenue_ttm: Mapped[Optional[int]] = mapped_column(Integer)
    gross_profit_ttm: Mapped[Optional[int]] = mapped_column(Integer)
    diluted_eps_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    quarterly_earnings_growth_yoy: Mapped[Optional[float]] = mapped_column(Integer)
    quarterly_revenue_growth_yoy: Mapped[Optional[float]] = mapped_column(Integer)
    analyst_target_price: Mapped[Optional[float]] = mapped_column(Integer)

    analyst_rating_strong_buy: Mapped[Optional[int]] = mapped_column(Integer)
    analyst_rating_buy: Mapped[Optional[int]] = mapped_column(Integer)
    analyst_rating_hold: Mapped[Optional[int]] = mapped_column(Integer)
    analyst_rating_sell: Mapped[Optional[int]] = mapped_column(Integer)
    analyst_rating_strong_sell: Mapped[Optional[int]] = mapped_column(Integer)

    trailing_pe: Mapped[Optional[float]] = mapped_column(Integer)
    forward_pe: Mapped[Optional[float]] = mapped_column(Integer)
    price_to_sales_ratio_ttm: Mapped[Optional[float]] = mapped_column(Integer)
    price_to_book_ratio: Mapped[Optional[float]] = mapped_column(Integer)
    ev_to_revenue: Mapped[Optional[float]] = mapped_column(Integer)
    ev_to_ebitda: Mapped[Optional[float]] = mapped_column(Integer)
    beta: Mapped[Optional[float]] = mapped_column(Integer)

    fifty_two_week_high: Mapped[Optional[float]] = mapped_column(Integer)
    fifty_two_week_low: Mapped[Optional[float]] = mapped_column(Integer)
    fifty_day_moving_average: Mapped[Optional[float]] = mapped_column(Integer)
    two_hundred_day_moving_average: Mapped[Optional[float]] = mapped_column(Integer)

    shares_outstanding: Mapped[Optional[int]] = mapped_column(Integer)
    dividend_date: Mapped[Optional[DateTime]] = mapped_column(DateTime)
    ex_dividend_date: Mapped[Optional[DateTime]] = mapped_column(DateTime)

    stock: Mapped["Stock"] = relationship(back_populates="metrics")
