from fastapi import APIRouter

from app.api.deps import SessionDep
from scrape.nasdaq import Scraper

router = APIRouter(prefix="/scrape", tags=["scrape"])

@router.get("/scrape/nasdaq")
async def get_nasdaq_stocks(session: SessionDep):
    scraper = Scraper(session=session)
    scraper.get_symbols()
    scraper.get_stocks()
