from copy import deepcopy
from typing import List

from sqlalchemy import select

from app.api.deps import SessionDep
from app.api.routes.stock import get_stock
from app.models import Stock
from app.schemas import StockRead, ChartRead
import pandas as pd


def normalize(stock: StockRead) -> StockRead:
    """
    Given a StockRead with irregular ChartRead dates (e.g. skipping weekends/holidays),
    return a new StockRead whose .charts cover every business day in the span,
    forward-filling prices/indicators and zeroing out volume on non-trading days.
    """
    # Make a deep copy so we don’t mutate the original
    stock = deepcopy(stock)

    # 1) Build a DataFrame from the existing charts
    df = pd.DataFrame([{
        "date": c.date,
        "symbol": c.symbol,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume,
        "adx_14": c.adx_14,
        "adx_120": c.adx_120,
        "dmi_positive_14": c.dmi_positive_14,
        "dmi_negative_14": c.dmi_negative_14,
        "dmi_positive_120": c.dmi_positive_120,
        "dmi_negative_120": c.dmi_negative_120,
        "rsi_14": c.rsi_14,
        "rsi_120": c.rsi_120,
    } for c in stock.charts])
    df.set_index("date", inplace=True)

    # 2) Create a business-day index from first to last date
    bdays = pd.bdate_range(df.index.min(), df.index.max())

    # 3) Reindex to include every business day
    df = df.reindex(bdays)

    # 4) Forward-fill the last known close
    df["close"] = df["close"].ffill()

    # 5) For missing days, use last close as open/high/low/close
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])

    # 6) Volume = 0 on non-trading days
    df["volume"] = df["volume"].fillna(0)

    # 7) Forward-fill all indicators
    for col in (
            "adx_14", "adx_120",
            "dmi_positive_14", "dmi_negative_14",
            "dmi_positive_120", "dmi_negative_120",
            "rsi_14", "rsi_120"
    ):
        df[col] = df[col].ffill()

    # 8) Rebuild stock.charts as ChartRead objects
    stock.charts = [
        ChartRead(
            date=dt,
            symbol=stock.symbol,
            open=int(row.open),
            high=int(row.high),
            low=int(row.low),
            close=int(row.close),
            volume=int(row.volume),
            adx_14=row.adx_14,
            adx_120=row.adx_120,
            dmi_positive_14=row.dmi_positive_14,
            dmi_negative_14=row.dmi_negative_14,
            dmi_positive_120=row.dmi_positive_120,
            dmi_negative_120=row.dmi_negative_120,
            rsi_14=row.rsi_14,
            rsi_120=row.rsi_120,
        )
        for dt, row in df.iterrows()
    ]

    return stock


class SgpLSTM:
    def __init__(self, session_dep: SessionDep):
        self.session = session_dep
        self.stocks: List[StockRead] = list()

    async def load_features(self):
        symbols = self.session.execute(select(Stock.symbol)).scalars().all()
        for symbol in symbols:
            stock = await get_stock(symbol)
            normalized_stock = normalize(stock)
            self.stocks.append(normalized_stock)
