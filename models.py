import uuid

from sqlalchemy import create_engine, MetaData, Table, Column, String, text, DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import UUID

metadata = MetaData()

# Table definition
stock = Table(
    "stocks",
    metadata,
    Column("uuid", UUID, primary_key=True, unique=True, default=uuid.uuid4, nullable=False),
    Column("symbol", String, unique=True),
    Column("name", String, nullable=False),
    Column("industry", String),
    Column("marketcap", DOUBLE_PRECISION, nullable=False),
    schema="public"
)

if __name__ == "__main__":
    # Replace with your connection details
    engine = create_engine("postgresql+psycopg2://stock:stockpass@127.0.0.1/StockAnalysis")

    # Create the database
    with engine.connect() as connection:
        connection.execute(text('CREATE DATABASE "StockAnalysis" OWNER "user";'))
