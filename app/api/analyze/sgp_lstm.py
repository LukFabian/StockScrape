from app.api.analyze.sgp_lstm_trainer import SgpLSTMTrainer
from app.api.deps import SessionDep
from app.models import Stock
from app.schemas import ClassificationPrediction
from sqlalchemy.orm import sessionmaker


async def analyze_sgp_lstm(stock: Stock, session: SessionDep, session_factory: sessionmaker) -> ClassificationPrediction:
    _trainer = SgpLSTMTrainer(session, session_factory)
    _trainer.train()

    prob, label = _trainer.predict(stock)
    return ClassificationPrediction(probability=prob, label=label)
