import logging
import os
import random
import time
import requests
import sys
import json
import pathlib

from database.manager import DatabaseManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger level to INFO

# Create a handler to write log messages to standard output
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)  # Ensure the handler level is set to INFO

# Define the logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stdout_handler)

file_path = pathlib.Path(__file__).parent.resolve()


def get_symbols() -> list:
    urls = list()
    for i in range(1, 3):
        urls.append(
            f"https://api.stockanalysis.com/api/screener/s/f?m=s&s=asc&c=s,n,industry,marketCap&cn=1000&p={i}&i=stocks")
    data = list()
    for i, url in enumerate(urls):
        logger.info(f"Fetching stock symbols from stockanalysis.com - Request: {i + 1} out of {len(urls)}")
        time.sleep(random.randint(1, 3))
        json_resp = requests.get(url).json()
        data.append(json_resp.get("data").get("data"))
    return data


class Scraper:
    base_url = "https://www.nasdaq.com/market-activity/stocks/"
    symbols_data: list[dict] | None = None

    def get_symbols(self):
        self.symbols_data = get_symbols()

    def save_symbols(self):
        pass


if __name__ == "__main__":
    alembic_config_path = pathlib.Path(file_path.parent.resolve(), "alembic.ini").resolve()
    db_url = os.environ["DB_URL"]
    if db_url == "":
        logger.critical("empty DB_URL environment variable")
    db_manager = DatabaseManager(os.environ["DB_URL"], str(alembic_config_path))
    conn = db_manager.get_connection()
    conn.close()
    scraper = Scraper()
    scraper.get_symbols()
    with open("data.json", "w") as json_file:
        json.dump(scraper.symbols_data, json_file, indent=4)
    db_manager.close()
