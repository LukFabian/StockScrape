from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, asc, cast, Numeric, desc
from sqlalchemy.orm import aliased

from app.api.deps import SessionDep
from app.models import Chart, Stock
from app.schemas import StockPerformanceRead, ChartRead
from financial_mathematics.average_directional_index import calculate_adx


def process_stocks_from_alphavantage(session: SessionDep, stock_data: dict, time_frame: str) -> Stock:
    meta_data = stock_data.get("Meta Data")
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
            charts = stock_data.get("Monthly Time Series")
        case "WEEKLY":
            charts = stock_data.get("Weekly Time Series")
        case "DAILY":
            charts = stock_data.get("Time Series (Daily)")
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


def calculate_technical_stock_data(session: SessionDep, symbol: str):
    # 1. Fetch high, low, close data ordered by date ASC (oldest first)
    stmt = (
        select(Chart.date, Chart.high, Chart.low, Chart.close)
        .where(Chart.symbol == symbol)
        .order_by(asc(Chart.date))
    )
    result = session.execute(stmt).all()
    if not result or len(result) < 28:
        raise ValueError(f"Not enough chart data for symbol '{symbol}' to compute ADX. Need at least 14 data points.")

    # 2. Unpack the data
    dates: List = [row.date for row in result]
    highs: List[int] = [row.high for row in result]
    lows: List[int] = [row.low for row in result]
    closes: List[int] = [row.close for row in result]

    # 3. Calculate ADX values
    adx_14_list = calculate_adx(highs, lows, closes, period=28)
    if len(result) >= 120:
        adx_120_list = calculate_adx(highs, lows, closes, period=120)

    # 4. Optional: attach the latest ADX value to the most recent chart row in DB
    latest_chart = session.get(Chart, {"symbol": symbol, "date": dates[-1]})
    if latest_chart:
        latest_chart.adx_14 = adx_14_list[-1]  # most recent ADX value
        latest_chart.adx_120 = adx_120_list[-1]
        session.commit()
    else:
        raise ValueError(f"No latest chart found for symbol '{symbol}' on date {dates[-1]}")


def get_stock_performance(session, stock_symbol: str, start_time: datetime, is_best: bool) -> Optional[
    StockPerformanceRead]:
    # Aliases for subqueries
    now_chart = aliased(Chart)
    past_chart = aliased(Chart)
    stock_now = aliased(Stock)
    stock_past = aliased(Stock)

    # Subquery for latest close_now
    now_subq = (
        select(
            stock_now.symbol.label("symbol"),
            now_chart.close.label("close_now"),
            stock_now.last_modified.label("last_modified")
        )
        .join(now_chart, now_chart.symbol == stock_now.symbol)
        .where(stock_now.symbol == stock_symbol)
        .order_by(now_chart.date.desc())
        .limit(1)
        .subquery()
    )

    # Subquery for past close_past
    past_subq = (
        select(
            stock_past.symbol.label("symbol"),
            past_chart.close.label("close_past"),
            stock_past.last_modified.label("last_modified")  # ✅ fixed here
        )
        .join(past_chart, past_chart.symbol == stock_past.symbol)
        .where(
            stock_past.symbol == stock_symbol,
            past_chart.date >= start_time
        )
        .order_by(past_chart.date.asc())
        .limit(1)
        .subquery()
    )

    # Final query combining both
    performance_query = (
        select(
            now_subq.c.symbol,
            ((now_subq.c.close_now - past_subq.c.close_past) / cast(past_subq.c.close_past, Numeric) * 100).label(
                "performance"),
            now_subq.c.last_modified
        )
        .select_from(now_subq.join(past_subq, now_subq.c.symbol == past_subq.c.symbol))
    ).subquery()

    # Select best or worst performer
    order_clause = desc(performance_query.c.performance) if is_best else asc(performance_query.c.performance)

    row = session.execute(
        select(
            performance_query.c.symbol,
            performance_query.c.performance,
            performance_query.c.last_modified
        ).order_by(order_clause)
    ).first()

    if row is None:
        return None
    charts: List[Chart] = session.execute(select(Chart).where(Chart.symbol == stock_symbol)).scalars().all()
    charts_read: List[ChartRead] = [ChartRead.model_validate(chart) for chart in charts]
    return StockPerformanceRead(
        symbol=row.symbol,
        last_modified=row.last_modified,
        performance=float(row.performance),
        charts=charts_read
    )
