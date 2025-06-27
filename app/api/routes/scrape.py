from fastapi import APIRouter

from app.api.deps import SessionDep
from scrape.nasdaq import Scraper

router = APIRouter(prefix="/scrape", tags=["scrape"])

@router.get("/scrape/nasdaq/stock")
async def get_nasdaq_stocks(session: SessionDep) -> list[str]:
    scraper = Scraper(session=session)
    scraper.get_symbols()
    return scraper.get_stock_and_charts()
