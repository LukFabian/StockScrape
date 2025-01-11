import datetime
import random
import time
import requests
import pathlib

import sqlalchemy.exc

import models
from database.manager import DatabaseManager
from sqlalchemy.sql import Insert, Select, Update

from stock_scrape_logger import logger

file_path = pathlib.Path(__file__).parent.resolve()


def format_to_bigint(currency: str) -> str:
    try:
        currency = float(currency.replace(",", "")) * 100
    except AttributeError:
        pass
    rounded_currency = round(currency)
    return str(rounded_currency)


def process_stocks(stock_data: dict, conn: sqlalchemy.Connection):
    symbol = stock_data.get("symbol")
    logged_integrity_error = False
    for chart_dict in stock_data.get("chart"):
        chart = chart_dict.get("z")
        insert_stmt = Insert(models.Chart).values(
            symbol=symbol,
            high=str(format_to_bigint(chart.get("high"))),
            low=str(format_to_bigint(chart.get("low"))),
            open=str(format_to_bigint(chart.get("open"))),
            close=str(format_to_bigint(chart.get("close"))),
            volume=str(chart.get("volume")).replace(",", "") + "00",
            date=chart.get("dateTime")
        )
        try:
            conn.execute(insert_stmt)
        except sqlalchemy.exc.IntegrityError:
            if not logged_integrity_error:
                logger.info(f"integrity error on stock: {symbol}, updating instead")
            logged_integrity_error = True
            conn.rollback()
            update_stmt = Update(models.Chart).where(
                models.Chart.symbol == symbol and models.Chart.date == chart.get("dateTime")).values(
                high=str(format_to_bigint(chart.get("high"))),
                low=str(format_to_bigint(chart.get("low"))),
                open=str(format_to_bigint(chart.get("open"))),
                close=str(format_to_bigint(chart.get("close"))),
                volume=str(chart.get("volume")).replace(",", "") + "00",
            )
            conn.execute(update_stmt)
        conn.commit()
    insert_stmt = Update(models.Stock).where(models.Stock.symbol == symbol).values(
        isNasdaq100=stock_data.get("isNasdaq100"),
        deltaIndicator=stock_data.get("deltaIndicator")
    )
    conn.execute(insert_stmt)
    conn.commit()


class Scraper:
    symbols_data: list[list[dict]] = None
    alembic_config_path = None
    db_url = None
    db_manager = None
    request_headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0"}

    def __init__(self, db_url: str):
        self.alembic_config_path = pathlib.Path(file_path.parent.resolve(), "alembic.ini").resolve()
        self.db_url = db_url
        if self.db_url == "":
            logger.critical("empty DB_URL environment variable")
        self.db_manager = DatabaseManager(self.db_url, self.alembic_config_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.db_manager is not None:
            self.db_manager.close()

    def scrape_symbols(self, urls_to_visit: int = 5):
        urls = list()
        for i in range(1, urls_to_visit + 1):
            urls.append(
                f"https://api.stockanalysis.com/api/screener/s/f?m=s&s=asc&c=s,n,industry,marketCap&cn=1000&p={i}&i=stocks")
        data = list()
        for i, url in enumerate(urls):
            logger.info(f"Fetching stock symbols from stockanalysis.com - Request: {i + 1} out of {len(urls)}")
            time.sleep(random.randint(1, urls_to_visit))
            json_resp = requests.get(url, headers=self.request_headers).json()
            data.append(json_resp.get("data").get("data"))
        self.symbols_data = data
        self._write_symbols()

    def _load_symbols(self, limit=10) -> list[str]:
        select_stmt = Select(models.Stock.symbol).limit(limit)
        with self.db_manager.get_connection() as conn:
            results = conn.execute(select_stmt).scalars().all()
            return list(results)

    def _write_symbols(self):
        if self.symbols_data is None:
            self.scrape_symbols()
        with self.db_manager.get_connection() as conn:
            for symbol_list in self.symbols_data:
                for symbol in symbol_list:
                    insert_stmt = Insert(models.Stock).values(
                        symbol=symbol.get("s"),
                        name=symbol.get("n"),
                        industry=symbol.get("industry"),
                        marketcap=format_to_bigint(symbol.get("marketCap"))
                    )
                    try:
                        conn.execute(insert_stmt)
                    except sqlalchemy.exc.IntegrityError as e:
                        logger.warning(f"integrity error: {e}")
                    conn.commit()

    def scrape_stocks(self, from_db=True, amount_stocks=10):
        if from_db:
            symbols = self._load_symbols(limit=amount_stocks)
        elif self.symbols_data is None:
            self.scrape_symbols()
            symbols = self._load_symbols(limit=amount_stocks)
        else:
            self._write_symbols()
            symbols = self._load_symbols(limit=amount_stocks)
        today = datetime.datetime.now()
        time_delta = datetime.timedelta(days=365 * 5)
        five_years_ago = today - time_delta
        formatting = "%Y-%m-%d"
        five_years_ago_str = five_years_ago.strftime(formatting)
        today_str = today.strftime(formatting)
        animation = "|/-\\"
        with self.db_manager.get_connection() as conn:
            for symbol in symbols:
                request_url = f"https://api.nasdaq.com/api/quote/{symbol.lower()}/chart?assetclass=stocks&fromdate={five_years_ago_str}&todate={today_str}"
                json_resp = requests.get(request_url, headers=self.request_headers).json()
                process_stocks(json_resp.get("data"), conn)
                idx = 0
                for i in range(5):
                    print(animation[idx % len(animation)], end="\r")
                    idx += 1
                    time.sleep(random.randint(1, 4) / 10)
            logger.info(f"Successfully fetched stock data for stock: {symbol} from nasdaq")
