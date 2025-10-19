from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load the model
with open("ProfitPredictor.pkl", "rb") as f:
    model = pickle.load(f)

# Example: if your model was trained with states one-hot encoded like this:
STATE_MAP = {
    "California": [1, 0, 0],
    "Florida": [0, 1, 0],
    "New York": [0, 0, 1]
}

@app.get("/")
def read_root():
    return {"message": "Profit Prediction Model is live."}

@app.post("/predict")
def predict(
    rdSpend: float,
    adSpend: float,
    marSpend: float,
    state: str
):
    # Encode the state
    state_encoded = STATE_MAP.get(state)
    if state_encoded is None:
        return {"error": f"Invalid state '{state}'. Choose from {list(STATE_MAP.keys())}"}

    # Combine all inputs
    input_data = np.array([[rdSpend, adSpend, marSpend] + state_encoded])

    # Predict
    prediction = model.predict(input_data)[0]

    return {"Predicted Profit": float(prediction)}
