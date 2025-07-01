import asyncio
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow
from gplearn.functions import make_function
from pandas._libs.tslibs.offsets import BDay
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, or_

from app.api.analyze.normalization import batch_normalize_all, chart_cols, balance_sheet_cols
from app.api.analyze.symbolic_transformer import CustomSymbolicTransformer
from app.api.deps import SessionDep
from sqlalchemy.orm import sessionmaker, selectinload
from app.api.routes.stock import get_stock
from app.logger import logger
from app.models import Stock, Chart
from app.schemas import StockRead


def build_feature_df(stock: Stock | StockRead) -> pd.DataFrame:
    df_chart = pd.DataFrame([
        {
            "symbol": c.symbol,
            "date": c.date,
            **{col: getattr(c, col, None) for col in chart_cols}
        }
        for c in stock.charts
    ])

    df_balance_sheet = pd.DataFrame([
        {
            "symbol": c.symbol,
            "period_ending": c.period_ending,
            **{col: getattr(c, col, None) for col in balance_sheet_cols}
        }
        for c in stock.balance_sheets
    ])

    df_chart["date"] = pd.to_datetime(df_chart["date"])
    df_balance_sheet["period_ending"] = pd.to_datetime(df_balance_sheet["period_ending"])

    df_chart["date"] = df_chart["date"].dt.tz_localize(None)
    df_balance_sheet["period_ending"] = df_balance_sheet["period_ending"].dt.tz_localize(None)

    # Sort both DataFrames by symbol and date (required for merge_asof)
    df_chart_sorted = df_chart.sort_values(["symbol", "date"])
    df_bs_sorted = df_balance_sheet.sort_values(["symbol", "period_ending"])

    # Perform the asof merge
    df = pd.merge_asof(
        df_chart_sorted,
        df_bs_sorted,
        by="symbol",
        left_on="date",
        right_on="period_ending",
        direction="backward"
    )

    df = df.dropna(axis=0)
    return df


def apply_rolling_features(x_sgp: np.ndarray, windows=(3, 5, 10, 20)):
    """
    Applies rolling window transformations to each SGP feature.
    """
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

    return np.array(X_seq, dtype="float32"), np.array(y_seq, dtype="float32")


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
            try:
                asyncio.run(get_stock(session, symbol, with_technicals=True))
                session.commit()
            except Exception as e:
                logger.warning(f"Could not load stock {symbol}: {e}")
                session.rollback()

    def get_tf_dataset(self, window=5, seq_len=15, batch_size=64):
        self.load_features()
        X, y_binary, y_return = self.prepare_training_data(window=window)
        X_scaled = self.scaler.fit_transform(X)

        self.cst = CustomSymbolicTransformer(
            generations=10,
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
        X_seq, y_seq = create_lstm_sequences(X_sgp_rolled, y_binary, seq_len=seq_len)

        def generator():
            for x, y in zip(X_seq, y_seq):
                yield x, y

        # Dynamically infer shape from the first sample
        example_x = X_seq[0]
        output_signature = (
            tensorflow.TensorSpec(shape=example_x.shape, dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32),
        )

        dataset = tensorflow.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tensorflow.data.AUTOTUNE)
        return dataset

    def load_features(self, max_workers: int = 16):
        symbols = self.session.execute(select(Stock.symbol)).scalars().all()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.calculate_and_commit, self.sessionmaker, symbol)
                for symbol in symbols
            ]
            for future in as_completed(futures):
                future.result()

    def prepare_training_data(self, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares training data from all loaded and normalized Stock instances.
        Returns:
            X         -> Combined feature matrix
            y_binary  -> Binary classification labels (1 if return > median, else 0)
            y_return  -> Raw future return values
        """

        X_all = []
        y_binary_all = []
        y_return_all = []

        # Normalize all stocks before building training data
        batch_normalize_all(self.session, self.sessionmaker, max_workers=8)

        stock_symbols = self.session.execute(select(Stock.symbol)).scalars().all()

        for symbol in stock_symbols:
            stock = self.session.execute(
                select(Stock)
                .options(selectinload(Stock.charts))
                .options(selectinload(Stock.balance_sheets))
                .where(Stock.symbol == symbol)
            ).scalar_one_or_none()

            if not stock or not stock.charts:
                continue

            df = build_feature_df(stock)

            if df.empty:
                continue

            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            if df.shape[0] <= window:
                continue

            # Compute future return and binary classification label
            df["future_close"] = df["close"].shift(-window)
            df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
            df.dropna(subset=["future_return"], inplace=True)

            if df.empty:
                continue

            median_return = df["future_return"].median()
            df["label"] = (df["future_return"] > median_return).astype(int)

            # Append feature matrix and labels
            X_all.append(df[chart_cols + balance_sheet_cols].values)
            y_binary_all.append(df["label"].values)
            y_return_all.append(df["future_return"].values)

        if not X_all:
            raise ValueError("No training data extracted. Check your stock inputs.")

        X = np.vstack(X_all)
        y_binary = np.concatenate(y_binary_all)
        y_return = np.concatenate(y_return_all)

        return X, y_binary, y_return

    def train(self, window: int = 5, seq_len: int = 15):
        dataset = self.get_tf_dataset(window=window, seq_len=seq_len)
        dataset_size = sum(1 for _ in dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)

        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size + val_size)

        # Peek one batch for shape
        input_shape = train_ds.element_spec[0].shape[1:]

        self.model = build_lstm_model(input_shape=input_shape)
        self.model.fit(train_ds, validation_data=val_ds, epochs=10)

        y_true, y_pred_probs = [], []
        for x_batch, y_batch in test_ds:
            probs = self.model.predict(x_batch).ravel()
            y_true.extend(y_batch.numpy())
            y_pred_probs.extend(probs)

        y_pred_classes = (np.array(y_pred_probs) > 0.5).astype(int)
        print(classification_report(y_true, y_pred_classes))
        print("ROC AUC:", roc_auc_score(y_true, y_pred_probs))

        self.trained = True

    def predict(self, stock: StockRead, seq_len: int = 15) -> Tuple[float, int]:
        if not self.trained or self.model is None or self.cst is None:
            raise RuntimeError("Model is not trained. Call `train()` first.")

        df = build_feature_df(stock)
        if df.empty:
            raise ValueError("Insufficient data to make a prediction.")

        X_input = self.scaler.transform(df[chart_cols + balance_sheet_cols].values)
        X_sgp = self.cst.transform(X_input)
        X_sgp_rolled = apply_rolling_features(X_sgp)
        X_seq, _ = create_lstm_sequences(X_sgp_rolled, np.zeros(len(X_sgp_rolled)), seq_len=seq_len)

        x_latest = X_seq[-1].reshape(1, *X_seq.shape[1:])
        prob = self.model.predict(x_latest).item()
        label = int(prob > 0.5)
        return prob, label
