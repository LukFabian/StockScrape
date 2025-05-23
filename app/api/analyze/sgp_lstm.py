from datetime import datetime
from app.api.analyze.sgp_lstm_trainer import SgpLSTMTrainer
from app.api.deps import SessionDep
from app.schemas import StockRead


async def analyze_sgp_lstm(stock: StockRead, start: datetime, session: SessionDep) -> StockRead:
    _trainer = SgpLSTMTrainer(session)
    await _trainer.train()

    prob, label = _trainer.predict(stock)

    print(f"Prediction probability: {prob:.4f} | label: {'↑ likely rise' if label else '↓ likely fall'}")
    stock.symbol += f" | Pred: {label} ({prob:.2f})"  # TEMP: show result in symbol
    return stock
