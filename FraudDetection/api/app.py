from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict

app = FastAPI()

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    data = transaction.dict()

    result = predict(data)

    return {"fraud_prediction": result}