import asyncio
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import tensorflow
from gplearn.functions import make_function
from pandas._libs.tslibs.offsets import BDay
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, or_

from app.api.analyze.symbolic_transformer import CustomSymbolicTransformer
from app.api.deps import SessionDep
from sqlalchemy.orm import sessionmaker, selectinload
from app.api.routes.stock import get_stock
from app.logger import logger
from app.models import Stock, Chart
from app.schemas import StockRead


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


# Safe division
def _safe_div(x1, x2):
    result = np.ones_like(x1, dtype=np.float64)  # default to 1.0
    mask = np.abs(x2) > 1e-6
    result[mask] = np.divide(x1[mask], x2[mask])
    return result


def normalize(stock: Stock) -> Stock:
    """
    Given a Stock with irregular Chart dates (e.g. skipping weekends/holidays),
    return a new Stock whose .charts cover every business day in the span,
    forward-filling prices/indicators and zeroing out volume on non-trading days.
    This mutates the stock.charts relationship in-place.
    """
    # 1) Work on a deep copy so if you need the original you still have it
    stock = deepcopy(stock)

    # 2) Build a DataFrame from existing charts
    df = pd.DataFrame([
        {
            "date": c.date,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "adx_14": c.adx_14,
            "adx_120": c.adx_120,
            "dmi_plus_14": c.dmi_plus_14,
            "dmi_minus_14": c.dmi_minus_14,
            "dmi_plus_120": c.dmi_plus_120,
            "dmi_minus_120": c.dmi_minus_120,
            "rsi_14": c.rsi_14,
            "rsi_120": c.rsi_120,
        }
        for c in stock.charts
    ])
    df.set_index("date", inplace=True)

    # 3) Create full business‐day index
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())

    # 4) Reindex to include every business day
    df = df.reindex(bdays)

    # 5) Fill symbol (constant), forward‐fill/zero‐fill columns
    df["symbol"] = stock.symbol
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    # 6) Forward‐fill indicators, then fill any remaining NaN with defaults
    defaults = {
        "adx_14": 0.0,      "adx_120": 0.0,
        "dmi_plus_14": 0.0, "dmi_minus_14": 0.0,
        "dmi_plus_120": 0.0,"dmi_minus_120": 0.0,
        "rsi_14": 50.0,     "rsi_120": 50.0,
    }
    for col, default in defaults.items():
        df[col] = df[col].ffill().fillna(default)

    # 7) (Optional) Sanity check
    if df.isna().any().any():
        raise ValueError("normalize(): NaNs remain after filling!")

    # 8) Write back into stock.charts
    #    First clear existing list (SQLAlchemy will DELETE-orphan on flush if you persist)
    stock.charts.clear()

    #    Then create a Chart for each row
    for idx, row in df.iterrows():
        stock.charts.append(
            Chart(
                date=idx.to_pydatetime(),
                symbol=stock.symbol,
                open=int(row["open"]),
                high=int(row["high"]),
                low=int(row["low"]),
                close=int(row["close"]),
                volume=int(row["volume"]),
                adx_14=float(row["adx_14"]),
                adx_120=float(row["adx_120"]),
                dmi_plus_14=float(row["dmi_plus_14"]),
                dmi_minus_14=float(row["dmi_minus_14"]),
                dmi_plus_120=float(row["dmi_plus_120"]),
                dmi_minus_120=float(row["dmi_minus_120"]),
                rsi_14=float(row["rsi_14"]),
                rsi_120=float(row["rsi_120"]),
            )
        )

    # 9) Update last_modified so you know when this was normalized
    stock.last_modified = datetime.now().date()

    return stock

def normalize_if_needed(session_factory: sessionmaker, symbol: str) -> Optional[str]:
    """Loads a single Stock by symbol, normalizes if incomplete, and returns the symbol if normalized."""
    session = session_factory()
    try:
        stock = session.execute(
            select(Stock)
            .options(selectinload(Stock.charts))
            .where(Stock.symbol == symbol)
        ).scalar_one_or_none()

        if not stock or not stock.charts:
            return None

        # Check for missing business days
        chart_dates = sorted(c.date for c in stock.charts)
        expected = set(pd.bdate_range(chart_dates[0], chart_dates[-1]).date)
        actual = set(chart_dates)

        if len(expected - actual) > 0:
            # missing days ⇒ normalize & merge
            normalized = normalize(stock)
            logger.info(f"Normalized {symbol}")
            session.merge(normalized)
            session.commit()
            return symbol
        return None
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def batch_normalize_all(session: SessionDep, session_factory: sessionmaker, max_workers: int = 16):
    """Normalize all stocks in parallel, up to max_workers threads."""
    # fetch all symbols up front in the main thread
    symbols = session.execute(select(Stock.symbol)).scalars().all()
    session.close()

    normalized = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(normalize_if_needed, session_factory, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                normalized.append(result)
    if len(normalized) > 0:
        print(f"Normalized {len(normalized)} stocks: {normalized}")



def create_lstm_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 15):
    """
    Turns flat features into sequences of length `seq_len`.
    Each X[i] is a sequence: [t-seq_len, ..., t-1]
    Each y[i] is the target at time t
    """
    X_seq = []
    y_seq = []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def split_dataset(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    return (
        X[:train_end], y[:train_end],  # train
        X[train_end:val_end], y[train_end:val_end],  # val
        X[val_end:], y[val_end:]  # test
    )


def build_lstm_model(input_shape):
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Input(shape=input_shape))
    model.add(tensorflow.keras.layers.LSTM(64, return_sequences=True))
    model.add(tensorflow.keras.layers.LSTM(32))
    model.add(tensorflow.keras.layers.Dense(16, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))  # binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


safe_div = make_function(function=_safe_div, name='div', arity=2)

function_set = ['add', 'sub', 'mul', safe_div, 'log', 'sqrt', 'neg', 'abs']


class SgpLSTMTrainer:
    def __init__(self, session: SessionDep, sessionmaker: sessionmaker):
        self.session = session
        self.model = None
        self.scaler = StandardScaler()
        self.cst = None
        self.feature_cols = None
        self.trained = False
        self.sessionmaker = sessionmaker

    def stock_needs_technicals(self, symbol: str) -> bool:
        """Checks if technical indicators are missing for the last business day."""
        return self.session.execute(
            select(Chart)
            .where(
                (Chart.symbol == symbol) &
                (Chart.date == (datetime.now().date() - BDay(1)).date())
            )
            .where(
                or_(
                    Chart.adx_14.is_(None),
                    Chart.rsi_14.is_(None),
                    Chart.dmi_plus_14.is_(None),
                    Chart.dmi_minus_14.is_(None),
                    Chart.adx_120.is_(None),
                    Chart.rsi_120.is_(None),
                    Chart.dmi_plus_120.is_(None),
                    Chart.dmi_minus_120.is_(None),
                )
            )
            .limit(1)
        ).scalar_one_or_none() is not None

    def calculate_and_commit(self, sessionmaker, symbol: str):
        with sessionmaker() as session:
            if not self.stock_needs_technicals(symbol):
                return
            logger.info(f"Stock needs technical analysis: {symbol}")
            try:
                asyncio.run(get_stock(session, symbol, with_technicals=True))
                session.commit()
            except Exception as e:
                logger.warning(f"Could not load stock {symbol}: {e}")
                session.rollback()

    def load_features(self, max_workers: int = 16):
        symbols = self.session.execute(select(Stock.symbol)).scalars().all()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.calculate_and_commit, self.sessionmaker, symbol)
                for symbol in symbols
            ]
            for future in as_completed(futures):
                future.result()

    def prepare_training_data(self, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepares training data from all loaded and normalized StockRead instances.
        Returns:
            X         -> Combined feature matrix
            y_binary  -> Binary classification labels
            y_return  -> Future return values
            feature_names -> Names of base features
        """
        feature_cols = [
            "close", "volume",
            "adx_14", "dmi_plus_14",
            "dmi_minus_14", "rsi_14"
        ]

        X_all = []
        y_binary_all = []
        y_return_all = []
        batch_normalize_all(self.session, self.sessionmaker, max_workers=8)
        stock_symbols = self.session.execute(select(Stock.symbol)).scalars().all()

        for symbol in stock_symbols:
            # Load one stock and its charts on-demand
            stock = self.session.execute(
                select(Stock)
                .options(selectinload(Stock.charts))
                .where(Stock.symbol == symbol)
            ).scalar_one_or_none()

            if not stock or not stock.charts:
                continue

            df = pd.DataFrame([{
                "date": c.date,
                "close": c.close,
                "volume": c.volume,
                "adx_14": c.adx_14,
                "dmi_plus_14": c.dmi_plus_14,
                "dmi_minus_14": c.dmi_minus_14,
                "rsi_14": c.rsi_14,
            } for c in stock.charts])

            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            if df.shape[0] <= window:
                continue

            df["future_close"] = df["close"].shift(-window)
            df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
            df.dropna(subset=["future_return"], inplace=True)

            if df.empty:
                continue

            median_return = df["future_return"].median()
            df["label"] = (df["future_return"] > median_return).astype(int)
            df = df.dropna(axis=0)

            X_all.append(df[feature_cols].values)
            y_binary_all.append(df["label"].values)
            y_return_all.append(df["future_return"].values)

        if not X_all:
            raise ValueError("No training data extracted. Check your stock inputs.")

        X = np.vstack(X_all)
        y_binary = np.concatenate(y_binary_all)
        y_return = np.concatenate(y_return_all)

        return X, y_binary, y_return, feature_cols

    def train(self, window: int = 5, seq_len: int = 15):
        self.load_features()
        X_train, y_binary, y_return, self.feature_cols = self.prepare_training_data(window=window)

        X_scaled = self.scaler.fit_transform(X_train)

        self.cst = CustomSymbolicTransformer(
            generations=20,
            population_size=1000,
            hall_of_fame=100,
            n_components=50,
            function_set=function_set,
            parsimony_coefficient=0.0005,
            max_samples=0.9,
            verbose=1,
            random_state=42,
            y_return=y_return
        )
        self.cst.fit(X_scaled, y_binary)
        X_sgp = self.cst.transform(X_scaled)
        X_sgp_rolled = apply_rolling_features(X_sgp)

        X_lstm, y_lstm = create_lstm_sequences(X_sgp_rolled, y_binary, seq_len=seq_len)
        X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X_lstm, y_lstm)

        self.model = build_lstm_model(input_shape=X_train.shape[1:])
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

        y_pred_probs = self.model.predict(X_test).ravel()
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        print(classification_report(y_test, y_pred_classes))
        print("ROC AUC:", roc_auc_score(y_test, y_pred_probs))

        self.trained = True

    def predict(self, stock: StockRead, seq_len: int = 15) -> Tuple[float, int]:
        if not self.trained or self.model is None or self.cst is None:
            raise RuntimeError("Model is not trained. Call `train()` first.")

        df = pd.DataFrame([{
            "close": c.close,
            "volume": c.volume,
            "adx_14": c.adx_14,
            "dmi_plus_14": c.dmi_plus_14,
            "dmi_minus_14": c.dmi_minus_14,
            "rsi_14": c.rsi_14,
        } for c in stock.charts])

        df = df.dropna()
        if df.empty or len(df) < seq_len:
            raise ValueError("Insufficient data to make a prediction.")

        X_input = self.scaler.transform(df[self.feature_cols].values)
        X_sgp = self.cst.transform(X_input)
        X_sgp_rolled = apply_rolling_features(X_sgp)
        X_seq, _ = create_lstm_sequences(X_sgp_rolled, np.zeros(len(X_sgp_rolled)), seq_len=seq_len)

        x_latest = X_seq[-1].reshape(1, *X_seq.shape[1:])
        prob = self.model.predict(x_latest).item()
        label = int(prob > 0.5)
        return prob, label
