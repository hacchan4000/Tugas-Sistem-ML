import pandas as pd
import numpy as np
import math

import mlflow
import mlflow.sklearn

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ==== Load dataset ====
df = pd.read_csv("aapl.us.txt_preprocessing.csv")

# Menggunakan fitur Close_norm
dataset = df["Close_norm"].values.reshape(-1, 1)

# ==== Windowing function ====
def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ==== Split Train/Test ====
training_data_len = math.ceil(len(dataset) * 0.8)

train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test,  y_test  = create_window(test_data, 60)

# ==== MLflow setup ====
mlflow.set_tracking_uri("mlruns")  # lokal, tidak perlu server
mlflow.set_experiment("stock_prediction")

mlflow.start_run():
    mlflow.autolog()
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.fit(X_train, y_train)

    # ==== Evaluation ====
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("RMSE:", rmse)

    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.log_model(model, "model")

