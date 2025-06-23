from app.api.analyze.sgp_lstm_trainer import SgpLSTMTrainer
from app.api.deps import SessionDep
from app.schemas import StockRead, ClassificationPrediction
from sqlalchemy.orm import sessionmaker


async def analyze_sgp_lstm(stock: StockRead, session: SessionDep, sessionmaker: sessionmaker) -> ClassificationPrediction:
    _trainer = SgpLSTMTrainer(session, sessionmaker)
    await _trainer.train()

    prob, label = _trainer.predict(stock)
    return ClassificationPrediction(probability=prob, label=label)
