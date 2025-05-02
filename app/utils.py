from datetime import datetime

from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Chart, Stock


def process_stocks_from_alphavantage(session: SessionDep, stock_data: dict, time_frame: str) -> Stock:
    meta_data = stock_data.get("Meta Data")
    symbol = meta_data.get("2. Symbol")

    # 1. Upsert Stock
    stock = session.get(Stock, symbol)
    if not stock:
        stock = Stock(symbol=symbol, last_modified=datetime.now())  # Name fallback to symbol
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
