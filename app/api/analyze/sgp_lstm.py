from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import numpy as np
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from app.api.analyze.symbolic_transformer import CustomSymbolicTransformer
from app.api.deps import SessionDep
from app.api.routes.stock import get_stock
from app.models import Stock
from app.schemas import StockRead, ChartRead
import pandas as pd


# Safe division
def _safe_div(x1, x2):
    result = np.ones_like(x1, dtype=np.float64)  # default to 1.0
    mask = np.abs(x2) > 1e-6
    result[mask] = np.divide(x1[mask], x2[mask])
    return result


safe_div = make_function(function=_safe_div, name='div', arity=2)

function_set = ['add', 'sub', 'mul', safe_div, 'log', 'sqrt', 'neg', 'abs']


def apply_rolling_features(x_sgp: np.ndarray, windows=(3, 5, 10, 20)):
    """
    Applies rolling window transformations to each SGP feature.
    """
    import pandas as pd
    features = []
    for w in windows:
        rolled = pd.DataFrame(x_sgp).rolling(window=w, min_periods=1).mean().values
        features.append(rolled)
    return np.concatenate(features, axis=1)


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
            stock = await get_stock(self.session, symbol, with_technicals=True)
            normalized_stock = normalize(stock)
            self.stocks.append(normalized_stock)

    def prepare_training_data(self, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepares training data from all loaded and normalized StockRead instances.

        Returns:
        - X: Combined features from all stocks
        - y_binary: 1 if future return > median, else 0
        - y_return: Actual future return values
        - feature_names: List of feature column names
        """
        feature_cols = [
            "close", "volume",
            "adx_14", "dmi_positive_14",
            "dmi_negative_14", "rsi_14"
        ]

        X_all = []
        y_binary_all = []
        y_return_all = []

        for stock in self.stocks:
            df = pd.DataFrame([{
                "date": c.date,
                "close": c.close,
                "volume": c.volume,
                "adx_14": c.adx_14,
                "dmi_positive_14": c.dmi_positive_14,
                "dmi_negative_14": c.dmi_negative_14,
                "rsi_14": c.rsi_14,
            } for c in stock.charts])

            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Skip if not enough data
            if df.shape[0] <= window:
                continue

            # Calculate future returns
            df["future_close"] = df["close"].shift(-window)
            df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
            df.dropna(subset=["future_return"], inplace=True)

            # Skip if no valid return data
            if df.empty:
                continue

            # Compute median per-stock
            median_return = df["future_return"].median()
            df["label"] = (df["future_return"] > median_return).astype(int)

            # Remove all nan-value rows
            df = df.dropna(axis=0)

            # Extract features and targets
            X_all.append(df[feature_cols].values)
            y_binary_all.append(df["label"].values)
            y_return_all.append(df["future_return"].values)

        # Combine all stocks into single training arrays
        if not X_all:
            raise ValueError("No training data extracted. Check your stock inputs.")

        X = np.vstack(X_all)
        y_binary = np.concatenate(y_binary_all)
        y_return = np.concatenate(y_return_all)

        return X, y_binary, y_return, feature_cols


async def analyze_sgp_lstm(stock: StockRead, start: datetime, session: SessionDep,
                           periods_to_predict: int = 14) -> StockRead:
    sgp_lstm = SgpLSTM(session)
    await sgp_lstm.load_features()
    X_train, y_binary, y_return, feature_names = sgp_lstm.prepare_training_data(window=5)
    # Now ready to pass into scaler + CustomSymbolicTransformer
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    cst = CustomSymbolicTransformer(
        generations=20,
        population_size=1000,
        hall_of_fame=100,
        n_components=50,
        function_set=['add', 'sub', 'mul', 'div', 'neg', 'sqrt', 'log'],
        parsimony_coefficient=0.0005,
        max_samples=0.9,
        verbose=1,
        random_state=42,
        y_return=y_return
    )

    cst.fit(X_scaled, y_binary)
    X_sgp = cst.transform(X_scaled)

    return StockRead(symbol="TSLA")  # todo: remove me
