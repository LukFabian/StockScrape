from sqlalchemy import create_engine, MetaData, Table, Column, String, text
from sqlalchemy.schema import CreateSchema

metadata = MetaData()

# Table definition
stock = Table(
    "stock",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("industry", String, nullable=False),
    schema="stockanalysis"
)

if __name__ == "__main__":
    # Replace with your connection details
    engine = create_engine("postgresql+psycopg2://stock:stockpass@127.0.0.1/StockAnalysis")

    # Create the database
    with engine.connect() as connection:
        connection.execute(text('CREATE DATABASE "StockAnalysis" OWNER "user";'))
