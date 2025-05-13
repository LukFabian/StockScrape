import pytz

from app.schemas import StockRead, ChartRead
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta, datetime


def analyze_linear_regression(stock: StockRead, start: datetime, periods_to_predict: int = 14) -> StockRead:
    if not stock.charts or len(stock.charts) < 2:
        # Not enough data to perform regression
        return stock

    # Convert dates to numeric values
    base_date = start if start else stock.charts[0].date

    X = np.array([(chart.date - base_date).days for chart in stock.charts]).reshape(-1, 1)
    y = np.array([chart.close for chart in stock.charts])

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future close prices
    last_day = (stock.charts[-1].date - base_date).days
    predictions = []
    for i in range(1, periods_to_predict + 1):
        future_day = last_day + i
        predicted_close = model.predict(np.array([[future_day]]))[0]

        # Create a new ChartRead with predicted data
        future_date = stock.charts[-1].date + timedelta(days=i)
        predicted_chart = ChartRead(
            date=future_date,
            symbol=stock.symbol,
            high=int(predicted_close),  # Simplified assumption
            low=int(predicted_close),   # Simplified assumption
            open=int(predicted_close),  # Simplified assumption
            close=int(predicted_close),
            volume=0,  # No volume data for predicted
        )
        predictions.append(predicted_chart)

    # Append predictions to the charts list
    stock.charts.extend(predictions)

    return stock
