import joblib
import pandas as pd
import numpy as np

from src.utils.logger import log_prediction

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_features = joblib.load("models/features.pkl")


def predict(data):

    # convert dict → dataframe
    df = pd.DataFrame([data])

    # apply same encoding used in training
    df = pd.get_dummies(df)

    # align with training columns
    df = df.reindex(columns=model_features, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)

    # predict
    prediction = model.predict(df_scaled)

    prediction_value = int(prediction[0])
    
    log_prediction(data, prediction_value)

    print("Prediction logged.")

    return prediction_value