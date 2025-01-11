from typing import List

from sqlalchemy import Column, String, Boolean, Double, BigInteger, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped

Base = declarative_base()


class Stock(Base):
    __tablename__ = "stock"
    symbol = Column(String, nullable=False, primary_key=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    marketcap = Column(Double, nullable=False)
    isNasdaq100 = Column(Boolean)
    deltaIndicator = Column(String)

    charts: Mapped[List["Chart"]] = relationship("Chart", back_populates="stock")


class Chart(Base):
    __tablename__ = "chart"
    high = Column(BigInteger, nullable=False)
    low = Column(BigInteger, nullable=False)
    open = Column(BigInteger, nullable=False)
    close = Column(BigInteger, nullable=False)
    volume = Column(BigInteger, nullable=False)
    date = Column(Date, primary_key=True)
    symbol = Column(String, ForeignKey("stock.symbol"), primary_key=True)

    stock = relationship("Stock", back_populates="charts")
