import os

from ml.analyzer import Analyzer
from scrape.scraper import Scraper

if __name__ == "__main__":
    # scrape
    with Scraper(os.environ["DB_URL"]) as scraper:
        scraper.scrape_stocks(from_db=False, amount_stocks=50)
    #with Analyzer(os.environ["DB_URL"]) as analyzer:
    #    analyzer.train()
