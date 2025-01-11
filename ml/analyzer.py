import datetime
import pathlib

from sqlalchemy import func

from financial_mathematics import average_directional_index
from database.manager import DatabaseManager
from stock_scrape_logger import logger
from models import Stock, Chart
from scipy.interpolate import CubicSpline

file_path = pathlib.Path(__file__).parent.resolve()


class Analyzer:
    alembic_config_path = None
    db_url = None
    db_manager = None
    charts_to_analyze = 10

    def __init__(self, db_url: str, charts_to_analyze: int = 10):
        self.alembic_config_path = pathlib.Path(file_path.parent.resolve(), "alembic.ini").resolve()
        self.db_url = db_url
        self.charts_to_analyze = charts_to_analyze
        if self.db_url == "":
            logger.critical("empty DB_URL environment variable")
        self.db_manager = DatabaseManager(self.db_url, self.alembic_config_path)
        self.interpolate_chart_data()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.db_manager is not None:
            self.db_manager.close()

    def interpolate_chart_data(self):

        with self.db_manager.get_session() as session:
            stocks_with_charts = (
                session.query(Stock)
                .join(Chart, Stock.symbol == Chart.symbol)  # Join with the related `charts` table
                .group_by(Stock.symbol, Chart.date)  # Group by `Stock` to count related rows
                .having(func.count(Stock.charts) > 0)  # Only include stocks with charts
                .order_by(Chart.date)
                .all()
            )
            for stock in stocks_with_charts:
                interpolated_data = []

                # Extract dates and values
                charts_sorted = sorted(stock.charts, key=lambda chart: chart.date)
                dates = [chart.date for chart in charts_sorted]
                high_values = [chart.high for chart in charts_sorted]
                low_values = [chart.low for chart in charts_sorted]
                open_values = [chart.open for chart in charts_sorted]
                close_values = [chart.close for chart in charts_sorted]
                volume_values = [chart.volume for chart in charts_sorted]

                # Find the full date range
                all_dates = [dates[0] + datetime.timedelta(days=i) for i in range((dates[-1] - dates[0]).days + 1)]
                date_indices = list(set([(date - dates[0]).days for date in dates]))
                full_indices = list()
                for i in range(date_indices[-1]):
                    full_indices.append(i)
                if len(all_dates) == full_indices:
                    continue

                # Step 3: Perform cubic spline interpolation for each attribute
                high_spline = CubicSpline(date_indices, high_values)
                low_spline = CubicSpline(date_indices, low_values)
                open_spline = CubicSpline(date_indices, open_values)
                close_spline = CubicSpline(date_indices, close_values)
                volume_spline = CubicSpline(date_indices, volume_values)

                # Interpolate values for missing dates
                for i in full_indices:
                    current_date = all_dates[i]
                    if current_date not in dates:  # Only for missing dates
                        interpolated_data.append(
                            Chart(
                                date=current_date,
                                symbol=stock.symbol,
                                high=int(high_spline(i)),
                                low=int(low_spline(i)),
                                open=int(open_spline(i)),
                                close=int(close_spline(i)),
                                volume=int(volume_spline(i)),
                            )
                        )

            # Step 4: Add interpolated data to the session and commit
            session.add_all(interpolated_data)
            session.commit()

    def train(self):
        with self.db_manager.get_session() as session:
            results = session.query(Stock, Chart).join(Chart, Stock.symbol == Chart.symbol).all()
            for stock, _ in results:
                print(stock)

    def _create_features(self):

        train_data.chart.adx_14 = average_directional_index.calculate_adx(train_data.chart.high,
                                                                               train_data.chart.low,
                                                                               train_data.chart.close, 14)
        train_data.chart.adx_120 = average_directional_index.calculate_adx(train_data.chart.high,
                                                                                train_data.chart.low,
                                                                                train_data.chart.close, 120)
        print(
            f"Date: {self.train_data.chart.date}, High: {self.train_data.chart.high} Adx_14: {self.train_data.chart.adx_14} Adx_120: {self.train_data.chart.adx_120}")
