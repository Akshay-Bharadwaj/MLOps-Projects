import os
import pandas as pd
import numpy as np
from datetime import datetime

LOG_FILE = os.path.join("logs", "predictions.csv")

def log_prediction(data, prediction):
    os.makedirs("logs", exist_ok=True)
    record = data.copy()
    record['prediction'] = prediction
    record['timestamp'] = datetime.now()

    df = pd.DataFrame([record])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', index=False, header=False)

