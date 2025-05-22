from datetime import datetime
from unittest import TestCase

import pandas as pd

from app.api.analyze.sgp_lstm import normalize
from app.schemas import ChartRead, StockRead


class Test(TestCase):
    def test_normalize_fills_missing_business_days(self):
        symbol = "AAPL"
        charts = [
            ChartRead(
                date=datetime(2023, 1, 2),  # Monday
                symbol=symbol,
                open=100,
                high=110,
                low=90,
                close=105,
                volume=1000,
            ),
            ChartRead(
                date=datetime(2023, 1, 4),  # Wednesday (skipping Tuesday)
                symbol=symbol,
                open=106,
                high=112,
                low=101,
                close=110,
                volume=1200,
            )
        ]

        stock = StockRead(symbol=symbol, charts=charts)
        normalized = normalize(stock)

        # Check dates
        expected_dates = pd.bdate_range("2023-01-02", "2023-01-04").tolist()
        actual_dates = [c.date for c in normalized.charts]
        assert actual_dates == expected_dates

        # Check Tuesday's (2023-01-03) filled values
        tuesday = next(c for c in normalized.charts if c.date == datetime(2023, 1, 3))
        assert tuesday.close == 105  # forward-filled from Monday
        assert tuesday.open == 105
        assert tuesday.high == 105
        assert tuesday.low == 105
        assert tuesday.volume == 0  # non-trading day

        # Check original dates are preserved correctly
        monday = normalized.charts[0]
        assert monday.volume == 1000
        wednesday = normalized.charts[-1]
        assert wednesday.volume == 1200
