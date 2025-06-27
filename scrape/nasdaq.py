import random
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import pathlib
# BDay is business day, not birthday
from pandas.tseries.offsets import BDay
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Stock, Chart, BalanceSheet
from stock_scrape_logger import logger
from scrape.mappings import _BALANCE_SHEET_MAPPINGS

file_path = pathlib.Path(__file__).parent.resolve()


def parse_price(s: str) -> int | None:
    """Convert strings like '$1,234,000' or '-$670,000' or '--' to int or None."""
    if not s or s.strip() in ("--",):
        return None
    # strip out $ , and handle parentheses if any
    neg = "(" in s or s.strip().startswith("-")
    clean = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        val = int(clean)
    except ValueError:
        # if it ever comes in as float, fallback
        val = int(float(clean))
    val = val * 100
    return -val if neg else val


def parse_volume(value: str) -> int | None:
    """Convert volume string like '2,859,893' to int. Returns None if 'N/A'."""
    try:
        return int(value.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def normalize_missing_volumes(rows: list[dict]) -> None:
    """Replace 'N/A' volumes with the average of adjacent valid volumes (forward and backward)."""
    volumes = [parse_volume(row["volume"]) for row in rows]

    for i, volume in enumerate(volumes):
        if volume is None:
            prev_val = next((volumes[j] for j in range(i - 1, -1, -1) if volumes[j] is not None), None)
            next_val = next((volumes[j] for j in range(i + 1, len(volumes)) if volumes[j] is not None), None)

            if prev_val is not None and next_val is not None:
                volumes[i] = (prev_val + next_val) // 2
            elif prev_val is not None:
                volumes[i] = prev_val
            elif next_val is not None:
                volumes[i] = next_val
            else:
                volumes[i] = 0  # fallback

    # Write back normalized volumes
    for i, row in enumerate(rows):
        row["volume"] = str(volumes[i])


class Scraper:
    chart_url = "https://api.nasdaq.com/api/quote/{}/historical?assetclass=stocks&fromdate={}&limit=1300&offset=0&todate={}"
    balance_sheet_url = "https://api.nasdaq.com/api/company/{}/financials?frequency=1"
    symbols_data: list[dict] | None = None
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0"
    }
    session: SessionDep | None = None

    def __init__(self, session: SessionDep):
        self.session = session

    def get_symbols(self):
        urls = list()
        for i in range(1, 3):
            urls.append(
                f"https://api.stockanalysis.com/api/screener/s/f?m=s&s=asc&c=s,n,industry,marketCap&cn=1000&p={i}&i=stocks")
        data = list()
        for i, url in enumerate(urls):
            logger.info(f"Fetching stock symbols from stockanalysis.com - Request: {i + 1} out of {len(urls)}")
            time.sleep(random.randint(1, 3))
            json_resp = requests.get(url, headers=self.headers).json()
            data.append(json_resp.get("data").get("data"))
        self.symbols_data = data
        return self.symbols_data

    def get_stock_and_charts(self) -> list[str]:
        scraped_symbols = []
        if not self.symbols_data:
            self.get_symbols()

        for symbol_list in self.symbols_data:
            for symbol_dict in symbol_list:
                symbol = symbol_dict.get("s")
                # Check if today's chart already exists
                existing_stock = self.session.get(Stock, symbol)
                if existing_stock and any(
                        chart.date == (datetime.now().date() - BDay(1)).date() for chart in existing_stock.charts):
                    logger.info(f"Skipping update for {symbol} — already up to date.")
                    continue
                url = self.chart_url.format(symbol, (datetime.now() - relativedelta(years=5)).date(),
                                            datetime.now().date())
                time.sleep(random.randint(3, 5))
                data = requests.get(url, headers=self.headers).json().get("data")
                print(data)
                if data and data.get("tradesTable"):
                    logger.info(
                        f"Received chart data: {len(data['tradesTable']['rows'])} for stock with symbol: {symbol}")
                    self.insert_stock_and_chart_data(data)
                    scraped_symbols.append(symbol)
                else:
                    logger.warning(f"No data received for {symbol} with url: {url}")

        return scraped_symbols

    def insert_stock_and_chart_data(self, data: dict):
        symbol = data["symbol"]
        rows = data["tradesTable"]["rows"]
        normalize_missing_volumes(rows)

        # Parse incoming chart data
        incoming_charts = {
            datetime.strptime(row["date"], "%m/%d/%Y").date(): Chart(
                symbol=symbol,
                date=datetime.strptime(row["date"], "%m/%d/%Y").date(),
                open=parse_price(row["open"]),
                close=parse_price(row["close"]),
                high=parse_price(row["high"]),
                low=parse_price(row["low"]),
                volume=parse_volume(row["volume"])
            )
            for row in rows
        }

        stock = self.session.get(Stock, symbol)
        if stock is None:
            # Create new stock and assign all charts
            stock = Stock(symbol=symbol, last_modified=datetime.now())
            self.session.add(stock)
            self.session.flush()  # Make sure stock.id is available if needed

        # Fetch all existing charts for the symbol at once (efficient)
        existing_charts = {
            chart.date: chart for chart in self.session.query(Chart)
            .filter_by(symbol=symbol)
            .filter(Chart.date.in_(incoming_charts.keys()))
            .all()
        }

        for date, new_chart in incoming_charts.items():
            if date in existing_charts:
                # Update existing chart
                existing = existing_charts[date]
                existing.open = new_chart.open
                existing.close = new_chart.close
                existing.high = new_chart.high
                existing.low = new_chart.low
                existing.volume = new_chart.volume
            else:
                # Insert new chart
                self.session.add(new_chart)

        stock.last_modified = datetime.now()
        self.session.commit()

    def get_balance_sheets(self) -> list[str]:
        scraped_sheet_symbols = []
        symbols = self.session.execute(select(Stock.symbol)).scalars().all()
        for symbol in symbols:
            url = self.balance_sheet_url.format("AAPL")
            time.sleep(random.randint(3, 5))
            data = requests.get(url, headers=self.headers).json().get("data")
            print(data)
            if data and data.get("incomeStatementTable"):
                logger.info(
                    f"Received balance sheet data: {len(data['incomeStatementTable']['rows'])} for stock with symbol: {symbol}")
                self.insert_balance_sheet_data(data)
                scraped_sheet_symbols.append(symbol)
            else:
                logger.warning(f"No data received for {symbol} with url: {url}")
            break
        return scraped_sheet_symbols

    def insert_balance_sheet_data(self, data: dict):
        """
        `data` should be the value of parsed_json["data"], containing keys:
          - "symbol": str
          - "balanceSheetTable": { "headers": {...}, "rows": [...] }
        """
        symbol = data["symbol"]
        bs_table = data["balanceSheetTable"]
        headers = bs_table["headers"]

        # pull out all period-ending dates in order (value2, value3, …)
        # note: headers["value1"] is the label "Period Ending:", skip that
        period_keys = sorted(k for k in headers if k.startswith("value") and k != "value1")
        # e.g. ["value2", "value3", …]
        period_dates = [
            datetime.strptime(headers[k], "%m/%d/%Y").date()
            for k in period_keys
        ]

        # for each period, start with the base dict
        records = [
            {"symbol": symbol, "period_ending": d}
            for d in period_dates
        ]

        # walk every row, map its label to our field name, and fill each period
        for row in bs_table["rows"]:
            label = row["value1"].strip()
            if label not in _BALANCE_SHEET_MAPPINGS:
                # skip headers like "Current Assets", "Long-Term Assets", etc.
                continue

            field = _BALANCE_SHEET_MAPPINGS[label]
            # for each period column, parse and assign
            for idx, key in enumerate(period_keys):
                raw = row.get(key, "")
                records[idx][field] = parse_price(raw)

        # bulk insert all periods
        for rec in records:
            bs = BalanceSheet(**rec)
            self.session.add(bs)

        # finally, commit (or flush, if you manage transaction outside)
        self.session.commit()
