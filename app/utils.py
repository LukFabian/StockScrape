from app.models import Chart, Stock


def format_to_bigint(currency: str) -> str:
    try:
        currency = float(currency.replace(",", "")) * 100
    except AttributeError:
        pass
    rounded_currency = round(currency)
    return str(rounded_currency)


def process_stocks_from_nasdaq(stock_data: dict) -> Stock:
    symbol = stock_data.get("symbol")
    charts = list()
    for chart_dict in stock_data.get("chart"):
        chart = chart_dict.get("z")
        charts.append(Chart(
            symbol=symbol,
            high=format_to_bigint(chart.get("high")),
            low=format_to_bigint(chart.get("low")),
            open=format_to_bigint(chart.get("open")),
            close=format_to_bigint(chart.get("close")),
            volume=str(chart.get("volume")).replace(",", "") + "00",
            date=chart.get("dateTime"),
        ))
    stock = Stock(
        symbol=stock_data.get("symbol"),
        name=stock_data.get("company"),
        industry=stock_data.get("industry"),
        isNasdaq100=stock_data.get("isNasdaq100"),
        deltaIndicator=stock_data.get("deltaIndicator")
    )
    if stock_data.get("marketCap"):
        stock.marketcap = float(format_to_bigint(stock_data.get("marketCap"))),
    stock.charts = charts
    return stock
