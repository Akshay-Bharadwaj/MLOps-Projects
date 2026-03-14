import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from src.data.data_loader import load_data, split_data
from src.features.feature_engineering import scale_features


DATA_PATH = "data/creditcard.csv"
EXPERIMENT_NAME = "Credit Card Fraud Detection"


def train():

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # load data
        df = load_data(DATA_PATH)

        # 🔹 Encode categorical variables
        df = pd.get_dummies(df, drop_first=True)

        # split data
        X_train, X_test, y_train, y_test = split_data(df)

        # feature scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        model.fit(X_train_scaled, y_train)

        os.makedirs("models", exist_ok=True)

        joblib.dump(model, "models/model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(X_train.columns.tolist(), "models/features.pkl")

        # predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # log parameters
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42
        })

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # log model
        mlflow.sklearn.log_model(model, name="model")

        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("models/scaler.pkl")
        mlflow.log_artifact("models/features.pkl")

        print("Training completed and model logged successfully.")
        print(f"Precision score: {precision}")
        print(f"\nRecall score: {recall}")
        print(f"\nF1 Score: {f1}")


if __name__ == "__main__":
    train()