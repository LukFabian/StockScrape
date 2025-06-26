from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import select, asc, cast, Numeric, desc
from sqlalchemy.orm import aliased
import numpy as np

setattr(np, "NaN", np.nan)
import pandas_ta as ta
from app.api.deps import SessionDep
from app.logger import logger
from app.models import Chart, Stock, StockMetric, StockProfile
from app.schemas import StockPerformanceRead, ChartRead, StockRead


def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def process_charts_from_alphavantage(session: SessionDep, chart_data: dict, time_frame: str) -> Stock | None:
    if chart_data.get('Error Message'):
        raise HTTPException(status_code=400, detail=chart_data['Error Message'])
    if not chart_data.get("Meta Data"):
        logger.warn(f"Rate Limit Exceeded: {chart_data.get("Information")}")
        return None
    meta_data = chart_data.get("Meta Data")
    symbol = meta_data.get("2. Symbol")

    # 1. Upsert Stock
    stock = session.get(Stock, symbol)
    if not stock:
        stock = Stock(symbol=symbol, last_modified=datetime.now())  # Name fallback to symbol
        session.add(stock)
    else:
        stock.last_modified = datetime.now()
        session.add(stock)

    # 2. Select chart series
    match time_frame:
        case "MONTHLY":
            charts = chart_data.get("Monthly Time Series")
        case "WEEKLY":
            charts = chart_data.get("Weekly Time Series")
        case "DAILY":
            charts = chart_data.get("Time Series (Daily)")
        case _:
            raise ValueError(f"Unknown time_frame: {time_frame}")

    # 3. Prepare chart dates
    chart_dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in charts.keys()]

    # 4. Query existing chart records in bulk (avoid per-row queries)
    existing_chart_keys = set(
        session.execute(
            select(Chart.date, Chart.symbol)
            .where(Chart.symbol == symbol)
            .where(Chart.date.in_(chart_dates))
        ).all()
    )

    # 5. Bulk insert/update loop
    for date_str, values in charts.items():
        chart_date = datetime.strptime(date_str, "%Y-%m-%d")

        # Convert prices to *cents* (e.g., 280.52 → 28052)
        open_price = int(float(values["1. open"]) * 100)
        high_price = int(float(values["2. high"]) * 100)
        low_price = int(float(values["3. low"]) * 100)
        close_price = int(float(values["4. close"]) * 100)
        volume = int(values["5. volume"])

        # Check if date part of aggregate key already exists -> update if it does
        key = (chart_date, symbol)
        is_in_chart_keys = False
        for existing_key in existing_chart_keys:
            if existing_key[0] == key[0].date():
                is_in_chart_keys = True
        if is_in_chart_keys:
            # UPDATE existing chart
            chart = session.get(Chart, {"date": chart_date, "symbol": symbol})
            chart.open = open_price
            chart.high = high_price
            chart.low = low_price
            chart.close = close_price
            chart.volume = volume
        else:
            # INSERT new chart
            chart = Chart(
                date=chart_date,
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
            session.add(chart)
    session.commit()
    return stock


def calculate_technical_stock_data(stock: StockRead | StockPerformanceRead,
                                   session: SessionDep) -> StockRead | StockPerformanceRead:
    if len(stock.charts) < 28:
        raise ValueError(
            f"Not enough chart data for symbol '{stock.symbol}' to compute indicators. Need at least 14 data points."
        )

    # Unpack and convert price data to float (decimal dollars)
    data = {
        "date": [chart.date for chart in stock.charts],
        "open": [chart.open / 100 for chart in stock.charts],
        "high": [chart.high / 100 for chart in stock.charts],
        "low": [chart.low / 100 for chart in stock.charts],
        "close": [chart.close / 100 for chart in stock.charts],
        "volume": [chart.volume for chart in stock.charts],
    }

    df = pd.DataFrame(data)

    # Compute 14-period indicators
    df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    dmi_14 = ta.dm(df["high"], df["low"], length=14)
    df["dmi_plus_14"] = dmi_14["DMP_14"]
    df["dmi_minus_14"] = dmi_14["DMN_14"]

    # Compute 120-period indicators if possible
    if len(df) >= 120:
        df["adx_120"] = ta.adx(df["high"], df["low"], df["close"], length=120)["ADX_120"]
        df["rsi_120"] = ta.rsi(df["close"], length=120)
        dmi_120 = ta.dm(df["high"], df["low"], length=120)
        df["dmi_plus_120"] = dmi_120["DMP_120"]
        df["dmi_minus_120"] = dmi_120["DMN_120"]
    else:
        df["adx_120"] = None
        df["rsi_120"] = None
        df["dmi_plus_120"] = None
        df["dmi_minus_120"] = None

    # Build ChartRead list
    chart_objects = []
    for _, row in df.iterrows():
        chart = Chart(
            symbol=stock.symbol,
            date=row["date"],
            open=int(row["open"] * 100),
            high=int(row["high"] * 100),
            low=int(row["low"] * 100),
            close=int(row["close"] * 100),
            volume=int(row["volume"]),
            adx_14=row["adx_14"] if pd.notna(row["adx_14"]) else None,
            adx_120=row["adx_120"] if pd.notna(row["adx_120"]) else None,
            rsi_14=row["rsi_14"] if pd.notna(row["rsi_14"]) else None,
            rsi_120=row["rsi_120"] if pd.notna(row["rsi_120"]) else None,
            dmi_plus_14=row["dmi_plus_14"] if pd.notna(row["dmi_plus_14"]) else None,
            dmi_minus_14=row["dmi_minus_14"] if pd.notna(row["dmi_minus_14"]) else None,
            dmi_plus_120=row["dmi_plus_120"] if pd.notna(row["dmi_plus_120"]) else None,
            dmi_minus_120=row["dmi_minus_120"] if pd.notna(row["dmi_minus_120"]) else None,
        )
        chart_objects.append(chart)

    # Optionally delete existing charts if you want to overwrite
    session.query(Chart).filter(Chart.symbol == stock.symbol).delete()

    session.add_all(chart_objects)
    logger.info(f"Calculated technicals for stock: {stock.symbol}")

    return stock


def process_stock_metadata_from_alphavantage(session: SessionDep, stock_data: dict):
    symbol = stock_data.get("Symbol")
    if not symbol:
        raise ValueError("Missing 'Symbol' in stock data")

    # Ensure the Stock record exists or create it
    stock = session.get(Stock, symbol)
    now = datetime.now()
    if not stock:
        stock = Stock(symbol=symbol, last_modified=now)
        session.add(stock)
    else:
        stock.last_modified = now

    # Update or create StockProfile
    profile = session.get(StockProfile, symbol)
    if not profile:
        profile = StockProfile(symbol=symbol)
        stock.profile = profile  # sets relationship
        session.add(profile)

    profile.asset_type = stock_data.get("AssetType", "")
    profile.name = stock_data.get("Name", "")
    profile.description = stock_data.get("Description", "")
    profile.cik = stock_data.get("CIK", "")
    profile.exchange = stock_data.get("Exchange", "")
    profile.currency = stock_data.get("Currency", "")
    profile.country = stock_data.get("Country", "")
    profile.sector = stock_data.get("Sector", "")
    profile.industry = stock_data.get("Industry", "")
    profile.address = stock_data.get("Address", "")
    profile.official_site = stock_data.get("OfficialSite", "")
    profile.fiscal_year_end = stock_data.get("FiscalYearEnd", "")

    # Create a new StockMetric entry (assumes new metric entry per call)
    metric = StockMetric(
        symbol=symbol,
        date=now,  # Timestamp for when data was collected
        latest_quarter=parse_date(stock_data.get("LatestQuarter")),
        market_capitalization=parse_int(stock_data.get("MarketCapitalization")),
        ebitda=parse_int(stock_data.get("EBITDA")),
        pe_ratio=parse_float(stock_data.get("PERatio")),
        peg_ratio=parse_float(stock_data.get("PEGRatio")),
        book_value=parse_float(stock_data.get("BookValue")),
        dividend_per_share=parse_float(stock_data.get("DividendPerShare")),
        dividend_yield=parse_float(stock_data.get("DividendYield")),
        eps=parse_float(stock_data.get("EPS")),
        revenue_per_share_ttm=parse_float(stock_data.get("RevenuePerShareTTM")),
        profit_margin=parse_float(stock_data.get("ProfitMargin")),
        operating_margin_ttm=parse_float(stock_data.get("OperatingMarginTTM")),
        return_on_assets_ttm=parse_float(stock_data.get("ReturnOnAssetsTTM")),
        return_on_equity_ttm=parse_float(stock_data.get("ReturnOnEquityTTM")),
        revenue_ttm=parse_int(stock_data.get("RevenueTTM")),
        gross_profit_ttm=parse_int(stock_data.get("GrossProfitTTM")),
        diluted_eps_ttm=parse_float(stock_data.get("DilutedEPSTTM")),
        quarterly_earnings_growth_yoy=parse_float(stock_data.get("QuarterlyEarningsGrowthYOY")),
        quarterly_revenue_growth_yoy=parse_float(stock_data.get("QuarterlyRevenueGrowthYOY")),
        analyst_target_price=parse_float(stock_data.get("AnalystTargetPrice")),
        analyst_rating_strong_buy=parse_int(stock_data.get("AnalystRatingStrongBuy")),
        analyst_rating_buy=parse_int(stock_data.get("AnalystRatingBuy")),
        analyst_rating_hold=parse_int(stock_data.get("AnalystRatingHold")),
        analyst_rating_sell=parse_int(stock_data.get("AnalystRatingSell")),
        analyst_rating_strong_sell=parse_int(stock_data.get("AnalystRatingStrongSell")),
        trailing_pe=parse_float(stock_data.get("TrailingPE")),
        forward_pe=parse_float(stock_data.get("ForwardPE")),
        price_to_sales_ratio_ttm=parse_float(stock_data.get("PriceToSalesRatioTTM")),
        price_to_book_ratio=parse_float(stock_data.get("PriceToBookRatio")),
        ev_to_revenue=parse_float(stock_data.get("EVToRevenue")),
        ev_to_ebitda=parse_float(stock_data.get("EVToEBITDA")),
        beta=parse_float(stock_data.get("Beta")),
        fifty_two_week_high=parse_float(stock_data.get("52WeekHigh")),
        fifty_two_week_low=parse_float(stock_data.get("52WeekLow")),
        fifty_day_moving_average=parse_float(stock_data.get("50DayMovingAverage")),
        two_hundred_day_moving_average=parse_float(stock_data.get("200DayMovingAverage")),
        shares_outstanding=parse_int(stock_data.get("SharesOutstanding")),
        dividend_date=parse_date(stock_data.get("DividendDate")),
        ex_dividend_date=parse_date(stock_data.get("ExDividendDate")),
    )

    stock.metrics.append(metric)
    session.commit()


def get_stocks_performance(session, stock_symbols: List[str], is_best: bool, start_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> List[StockPerformanceRead]:
    # Aliases
    now_chart = aliased(Chart)
    past_chart = aliased(Chart)
    stock_now = aliased(Stock)
    stock_past = aliased(Stock)

    # Subquery: Latest close price (now)
    now_subq = (
        select(
            stock_now.symbol.label("symbol"),
            now_chart.close.label("close_now"),
            stock_now.last_modified.label("last_modified")
        )
        .join(now_chart, now_chart.symbol == stock_now.symbol)
        .where(stock_now.symbol.in_(stock_symbols))
        .order_by(stock_now.symbol, now_chart.date.desc())
        .distinct(stock_now.symbol)
        .subquery()
    )

    # Subquery: Earliest close price (past)
    if start_time:
        past_subq = (
            select(
                stock_past.symbol.label("symbol"),
                past_chart.close.label("close_past"),
                stock_past.last_modified.label("last_modified")
            )
            .join(past_chart, past_chart.symbol == stock_past.symbol)
            .where(
                stock_past.symbol.in_(stock_symbols),
                past_chart.date >= start_time
            )
            .order_by(stock_past.symbol, past_chart.date.asc())
            .distinct(stock_past.symbol)
            .subquery()
        )
    else:
        past_subq = (
            select(
                stock_past.symbol.label("symbol"),
                past_chart.close.label("close_past"),
                stock_past.last_modified.label("last_modified")
            )
            .join(past_chart, past_chart.symbol == stock_past.symbol)
            .where(
                stock_past.symbol.in_(stock_symbols)
            )
            .order_by(stock_past.symbol, past_chart.date.asc())
            .distinct(stock_past.symbol)
            .subquery()
        )

    # Combine and compute performance
    performance_query = (
        select(
            now_subq.c.symbol,
            ((now_subq.c.close_now - past_subq.c.close_past) / cast(past_subq.c.close_past, Numeric) * 100).label(
                "performance"),
            now_subq.c.last_modified
        )
        .select_from(now_subq.join(past_subq, now_subq.c.symbol == past_subq.c.symbol))
    )

    order_clause = desc("performance") if is_best else asc("performance")
    if limit:
        performance_query = performance_query.order_by(order_clause).limit(limit)
    else:
        performance_query = performance_query.order_by(order_clause)

    rows = session.execute(performance_query).all()

    result = []
    for row in rows:
        charts: List[Chart] = session.execute(
            select(Chart).where(Chart.symbol == row.symbol).order_by(Chart.date)
        ).scalars().all()
        charts_read: List[ChartRead] = [ChartRead.model_validate(chart) for chart in charts]

        result.append(
            StockPerformanceRead(
                symbol=row.symbol,
                last_modified=row.last_modified,
                performance=float(row.performance),
                charts=charts_read
            )
        )

    return result
