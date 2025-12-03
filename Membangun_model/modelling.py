import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR

df = pd.read_csv("Membangun_model/aapl.us.txt_preprocessing.csv")
dataset = df["Close_norm"].values.reshape(-1, 1)

def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

training_data_len = math.ceil(len(dataset) * 0.8)
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test, y_test = create_window(test_data, 60)


mlflow.set_experiment('stock_prediction')
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.autolog()

# TIDAK ADA start_run() DI SINI
model = SVR(kernel="rbf", C=100, gamma=0.1)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
mlflow.log_metric("test_score", float(score))

# Explicitly log model artifact (autolog often does this; this ensures a named artifact path)
mlflow.sklearn.log_model(model, artifact_path="model")

# Print helpful info for serving

print("done")
