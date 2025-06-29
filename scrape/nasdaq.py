import random
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import pathlib

from sqlalchemy import select

from app.api.deps import SessionDep
from app.models import Stock, Chart, BalanceSheet
from app.utils import flatten
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

    def get_symbols(self, skip_existing_symbols: bool = True) -> list[dict]:
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
        data = flatten(data)
        if skip_existing_symbols:
            symbols = self.session.execute(select(Stock.symbol)).scalars().all()
            filtered_data = list()
            for i, symbol_dict in enumerate(data):
                symbol = symbol_dict["s"]
                if symbol not in symbols:
                    filtered_data.append(symbol_dict)
            data = filtered_data
        self.symbols_data = data
        return self.symbols_data

    def get_stock_and_charts(self) -> list[str]:
        scraped = []

        if not self.symbols_data:
            self.get_symbols()

        for sym_d in self.symbols_data:
            symbol = sym_d.get("s")

            # 1. Fetch BS JSON (but don't insert yet)
            time.sleep(random.randint(2, 4))
            bs_payload = (
                requests.get(self.balance_sheet_url.format(symbol), headers=self.headers)
                .json()
                .get("data")
            )
            if not bs_payload or not bs_payload.get("incomeStatementTable"):
                logger.info(f"Skipping {symbol} — no balance-sheet data.")
                continue

            # 2. Fetch Chart JSON
            start = (datetime.now() - relativedelta(years=5)).date()
            end = datetime.now().date()
            time.sleep(random.randint(2, 4))
            res = requests.get(self.chart_url.format(symbol, start, end), headers=self.headers)
            if res.status_code != 200:
                print("Bad request: ", res)
                continue
            chart_payload = (
                res.json().get("data")
            )
            if not chart_payload or not chart_payload.get("tradesTable"):
                logger.warning(f"Skipping {symbol} — no chart data.")
                continue

            # 3. Insert Stock + both BS & Charts in one go
            logger.info(f"Persisting {symbol}: balance sheets + charts")
            self._upsert_stock_bs_and_charts(symbol, bs_payload, chart_payload)
            scraped.append(symbol)

        return scraped

    def _upsert_stock_bs_and_charts(self, symbol: str, bs_data: dict, chart_data: dict):
        """Atomically create/update Stock, its BalanceSheets, and its Charts."""

        # --- 1) Ensure Stock exists ---
        stock = self.session.get(Stock, symbol)
        if stock is None:
            stock = Stock(symbol=symbol, last_modified=datetime.now())
            self.session.add(stock)
            # flush so that foreign-key relationships can be established
            self.session.flush()

        # --- 2) Build BalanceSheet objects in memory ---
        bs_table = bs_data["balanceSheetTable"]
        headers = bs_table["headers"]
        period_keys = sorted(k for k in headers if k.startswith("value") and k != "value1")

        period_dates = []
        for key in period_keys:
            try:
                d = datetime.strptime(headers[key], "%m/%d/%Y").date()
            except ValueError:
                # skip if header isn't a date
                continue
            period_dates.append((key, d))

        # step B: start a list of dicts (one per date)
        bs_records = [
            {"symbol": symbol, "period_ending": d}
            for _, d in period_dates
        ]

        # step C: iterate rows, map labels → field names, then fill each date’s slot
        for row in bs_table["rows"]:
            label = row["value1"].strip()
            if label not in _BALANCE_SHEET_MAPPINGS:
                continue
            field = _BALANCE_SHEET_MAPPINGS[label]
            for idx, (hdr_key, _) in enumerate(period_dates):
                raw = row.get(hdr_key, "")
                bs_records[idx][field] = parse_price(raw)

        # now turn into ORM objects
        bs_objs = [BalanceSheet(**rec) for rec in bs_records]

        # --- 3) Upsert Charts similarly ---
        rows = chart_data["tradesTable"]["rows"]
        normalize_missing_volumes(rows)

        incoming_charts = {
            datetime.strptime(r["date"], "%m/%d/%Y").date(): Chart(
                symbol=symbol,
                date=datetime.strptime(r["date"], "%m/%d/%Y").date(),
                open=parse_price(r["open"]),
                close=parse_price(r["close"]),
                high=parse_price(r["high"]),
                low=parse_price(r["low"]),
                volume=parse_volume(r["volume"]),
            )
            for r in rows
        }
        existing_charts = {
            c.date: c for c in (
                self.session.query(Chart)
                .filter_by(symbol=symbol)
                .filter(Chart.date.in_(incoming_charts.keys()))
                .all()
            )
        }

        for bs in bs_objs:
            self.session.add(bs)

        for dt, new_chart in incoming_charts.items():
            if dt in existing_charts:
                old = existing_charts[dt]
                old.open, old.high, old.low, old.close, old.volume = (
                    new_chart.open, new_chart.high, new_chart.low,
                    new_chart.close, new_chart.volume
                )
            else:
                self.session.add(new_chart)

        # --- 5) Finalize Stock & commit ---
        stock.last_modified = datetime.now()
        self.session.commit()
