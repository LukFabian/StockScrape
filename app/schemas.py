from typing import Optional, List
from datetime import datetime, date
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
    dmi_plus_14: Optional[float] = None
    dmi_minus_14: Optional[float] = None
    dmi_plus_120: Optional[float] = None
    dmi_minus_120: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_120: Optional[float] = None

    class Config:
        from_attributes = True


class StockProfileRead(BaseModel):
    asset_type: str
    name: str
    description: str
    cik: str
    exchange: str
    currency: str
    country: str
    sector: str
    industry: str
    address: str
    official_site: str
    fiscal_year_end: str

    class Config:
        from_attributes = True


class StockMetricRead(BaseModel):
    date: datetime
    latest_quarter: Optional[datetime]
    market_capitalization: Optional[int]
    ebitda: Optional[int]
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    book_value: Optional[float]
    dividend_per_share: Optional[float]
    dividend_yield: Optional[float]
    eps: Optional[float]
    revenue_per_share_ttm: Optional[float]
    profit_margin: Optional[float]
    operating_margin_ttm: Optional[float]
    return_on_assets_ttm: Optional[float]
    return_on_equity_ttm: Optional[float]
    revenue_ttm: Optional[int]
    gross_profit_ttm: Optional[int]
    diluted_eps_ttm: Optional[float]
    quarterly_earnings_growth_yoy: Optional[float]
    quarterly_revenue_growth_yoy: Optional[float]
    analyst_target_price: Optional[float]

    analyst_rating_strong_buy: Optional[int]
    analyst_rating_buy: Optional[int]
    analyst_rating_hold: Optional[int]
    analyst_rating_sell: Optional[int]
    analyst_rating_strong_sell: Optional[int]

    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    price_to_sales_ratio_ttm: Optional[float]
    price_to_book_ratio: Optional[float]
    ev_to_revenue: Optional[float]
    ev_to_ebitda: Optional[float]
    beta: Optional[float]

    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    fifty_day_moving_average: Optional[float]
    two_hundred_day_moving_average: Optional[float]

    shares_outstanding: Optional[int]
    dividend_date: Optional[datetime]
    ex_dividend_date: Optional[datetime]

    class Config:
        from_attributes = True


class StockRead(BaseModel):
    symbol: str
    last_modified: datetime = datetime.now().date()
    charts: List[ChartRead] = []
    profile: Optional[StockProfileRead] = None
    metrics: List[StockMetricRead] = []

    class Config:
        from_attributes = True


class StockPerformanceRead(StockRead):
    performance: float

class ClassificationPrediction(BaseModel):
    label: int
    probability: float


class BalanceSheetBase(BaseModel):
    symbol: str
    period_ending: date
    cash_and_cash_equivalents: Optional[int]
    short_term_investments: Optional[int]
    net_receivables: Optional[int]
    inventory: Optional[int]
    other_current_assets: Optional[int]
    total_current_assets: Optional[int]
    fixed_assets: Optional[int]
    goodwill: Optional[int]
    intangible_assets: Optional[int]
    other_assets: Optional[int]
    deferred_asset_charges: Optional[int]
    total_assets: Optional[int]
    accounts_payable: Optional[int]
    short_term_debt: Optional[int]
    other_current_liabilities: Optional[int]
    total_current_liabilities: Optional[int]
    long_term_debt: Optional[int]
    other_liabilities: Optional[int]
    deferred_liability_charges: Optional[int]
    misc_stocks: Optional[int]
    minority_interest: Optional[int]
    total_liabilities: Optional[int]
    common_stocks: Optional[int]
    capital_surplus: Optional[int]
    treasury_stock: Optional[int]
    other_equity: Optional[int]
    total_equity: Optional[int]
    total_liabilities_and_equity: Optional[int]

    class Config:
        from_attributes = True
