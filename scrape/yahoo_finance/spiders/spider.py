from time import sleep

import scrapy
import logging

from scrapy_playwright.page import PageMethod


class YahooFinanceSpider(scrapy.Spider):
    name = 'yahoo_finance'
    allowed_domains = ['yahoo.com']
    start_urls = ["https://finance.yahoo.com/quote/MSFT/history/",
                  "https://finance.yahoo.com/quote/MSFT/key-statistics/"]
    session = None

    def start_requests(self):
        yield scrapy.Request(url=self.start_urls[0], meta={
                "playwright": True,
                "playwright_page_methods": [
                    PageMethod("wait_for_selector", "xpath=//button[text()='Alle akzeptieren']"),
                    PageMethod("click", "xpath=//button[text()='Alle akzeptieren']"),
                    PageMethod("wait_for_timeout", 2000),  # wait to ensure JS updates are complete
                    PageMethod("pause"),  # This will pause and open the Playwright Inspector
                ],
                "playwright_context": "default",
            }, callback=self.parse)

    def parse(self, response, **kwargs):
        # this should return true if we got historical stock data
        logging.info(f"Spider: {self.name} visiting url: {response.url}")
        if response.url.endswith("/history/"):
            table_headings_data = response.xpath(
                "//div[@data-testid='history-table']/div[starts-with(@class, 'table-container')]/table/thead/tr/th[*]")
            table_content_data = response.xpath(
                "//div[@data-testid='history-table']/div[starts-with(@class, 'table-container')]/table/tbody/tr[*]")
            table_headings = table_headings_data.extract()
            table_content = table_content_data.extract()
            print(table_headings)
            print(table_content)
        sleep(10)

        next_links = response.xpath("//a[starts-with(@href, '/')]").extract()
        for link in next_links:
            yield response.follow(link, callback=self.parse)
