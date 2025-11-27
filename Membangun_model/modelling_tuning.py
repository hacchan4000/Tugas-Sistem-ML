import pandas as pd
import numpy as np
import math

import mlflow
import mlflow.keras
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

# ==== Load dataset ====
df = pd.read_csv("aapl.us.txt_preprocessing.csv")

dataset = df["Close_norm"].values.reshape(-1, 1)

# ==== Windowing ====
def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

training_data_len = math.ceil(len(dataset) * 0.8)

train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data)
X_test,  y_test  = create_window(test_data)

# === Reshape for LSTM ===
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ==== MLflow setup ====
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("stock_prediction_lstm")

with mlflow.start_run():
    mlflow.autolog()

    # ==== Build LSTM model ====
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # ==== Train ====
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # ==== Evaluation ====
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("RMSE:", rmse)

    mlflow.log_metric("RMSE", rmse)

    # Log model (Keras)
    mlflow.keras.log_model(model, "model")
