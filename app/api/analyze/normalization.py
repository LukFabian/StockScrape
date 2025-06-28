from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import select, inspect, exists, or_

from app.api.deps import SessionDep
from sqlalchemy.orm import sessionmaker, selectinload
from app.logger import logger
from app.models import Stock, Chart, BalanceSheet

chart_cols = [
    col.key for col in inspect(Chart).columns
                   if col.key not in ("date", "symbol")
]
balance_sheet_cols = [
                   col.key for col in inspect(BalanceSheet).columns
                   if col.key not in ("id", "symbol", "period_ending")
               ]


def normalize_if_needed(session_factory: sessionmaker, symbol: str) -> Optional[str]:
    """Loads a single Stock by symbol, normalizes if incomplete (missing dates or NULLs), and returns the symbol if normalized."""
    session = session_factory()
    try:
        # Preload Stock and charts
        stock = session.execute(
            select(Stock)
            .options(
                selectinload(Stock.charts),
                selectinload(Stock.balance_sheets)
            )
            .where(Stock.symbol == symbol)
        ).scalar_one_or_none()

        if not stock or not stock.charts:
            return None

        # Check for missing business days
        chart_dates = sorted(c.date for c in stock.charts)
        expected = set(pd.bdate_range(chart_dates[0], chart_dates[-1]).date)
        actual = set(d for d in chart_dates)
        missing_dates = expected - actual

        # Check for NULL values in DB using EXISTS
        has_nulls = (
            session.query(
                exists().where(
                    Chart.symbol == symbol
                ).where(
                    or_(
                        Chart.adx_14.is_(None),
                        Chart.adx_120.is_(None),
                        Chart.dmi_plus_14.is_(None),
                        Chart.dmi_minus_14.is_(None),
                        Chart.rsi_14.is_(None),
                        Chart.rsi_120.is_(None),
                    )
                )
            ).scalar()
            or
            session.query(
                exists().where(
                    BalanceSheet.symbol == symbol
                ).where(
                    or_(
                        BalanceSheet.total_assets.is_(None),
                        BalanceSheet.total_liabilities.is_(None),
                        BalanceSheet.total_equity.is_(None),
                    )
                )
            ).scalar()
        )

        if missing_dates or has_nulls:
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


def normalize(stock: Stock) -> Stock:
    """
    Normalize chart data and attach forward-filled balance sheet data.
    Fills every business day in the span with inferred data.
    """
    stock = deepcopy(stock)

    # Chart DataFrame
    df_charts = pd.DataFrame([
        {
            "date": c.date,
            **{col: getattr(c, col) for col in chart_cols}
        }
        for c in stock.charts
    ])
    df_charts.set_index("date", inplace=True)

    # Business days index
    bdays = pd.bdate_range(start=df_charts.index.min(), end=df_charts.index.max())
    df = df_charts.reindex(bdays)

    # Fill symbol
    df["symbol"] = stock.symbol

    # Fill core prices
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    # Fill technical indicators
    defaults = {
        "adx_14": 0.0, "adx_120": 0.0,
        "dmi_plus_14": 0.0, "dmi_minus_14": 0.0,
        "dmi_plus_120": 0.0, "dmi_minus_120": 0.0,
        "rsi_14": 50.0, "rsi_120": 50.0,
    }
    for col, default in defaults.items():
        df[col] = df[col].ffill().fillna(default)

    # Balance sheet merge (if exists)
    bs = stock.balance_sheets
    df_bs = pd.DataFrame([
        {
            **{k: getattr(b, k) for k in b.__table__.columns.keys()}
        }
        for b in bs
    ]).sort_values("period_ending")

    df_bs["period_ending"] = pd.to_datetime(df_bs["period_ending"])

    # Fill BS NaNs with 0
    bs_cols = [col for col in df_bs.columns if col != "period_ending"]
    df_bs[bs_cols] = df_bs[bs_cols].fillna(0)

    balance_sheets = list()
    for _, row in df_bs.iterrows():
        balance_sheets.append(BalanceSheet(**row[bs_cols].to_dict()))
    stock.balance_sheets = balance_sheets
    df = df.reset_index().rename(columns={"index": "date"})
    # Safety check
    if df.isna().any().any() or df_bs.empty and df_bs.isna().any().any():
        raise ValueError("normalize(): NaNs remain after filling!")

    # Clear existing and rebuild stock.charts
    stock.charts.clear()

    for _, row in df.iterrows():
        chart_kwargs = {
            "date": row["date"].to_pydatetime(),
            "symbol": stock.symbol,
            "open": int(row["open"]),
            "high": int(row["high"]),
            "low": int(row["low"]),
            "close": int(row["close"]),
            "volume": int(row["volume"]),
            "adx_14": float(row["adx_14"]),
            "adx_120": float(row["adx_120"]),
            "dmi_plus_14": float(row["dmi_plus_14"]),
            "dmi_minus_14": float(row["dmi_minus_14"]),
            "dmi_plus_120": float(row["dmi_plus_120"]),
            "dmi_minus_120": float(row["dmi_minus_120"]),
            "rsi_14": float(row["rsi_14"]),
            "rsi_120": float(row["rsi_120"]),
        }

        stock.charts.append(Chart(**chart_kwargs))
    stock.last_modified = datetime.now().date()
    return stock
