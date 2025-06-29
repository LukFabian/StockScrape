from typing import Optional

from fastapi import APIRouter, Query

from app.api.deps import SessionDep
from scrape.nasdaq import Scraper

router = APIRouter(prefix="/scrape", tags=["scrape"])

@router.get("/scrape/nasdaq/stock")
async def get_nasdaq_stocks(session: SessionDep, skip_existing_symbols: Optional[bool] = Query(True,
                                                            description="Skip currently existing symbols")) -> list[str]:
    scraper = Scraper(session=session)
    scraper.get_symbols(skip_existing_symbols=skip_existing_symbols)
    return scraper.get_stock_and_charts()
