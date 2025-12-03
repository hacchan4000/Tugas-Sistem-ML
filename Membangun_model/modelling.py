import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from mlflow.models.signature import infer_signature

df = pd.read_csv("Membangun_model/aapl.us.txt_preprocessing.csv")

dataset = df["Close_norm"].values.reshape(-1, 1)

def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

training_data_len = int(len(dataset) * 0.8)
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test, y_test = create_window(test_data, 60)

mlflow.set_experiment("stock_prediction")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model = SVR(kernel="rbf", C=100, gamma=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------------------
# 1. Buat signature dari training set
# -------------------------------------------
signature = infer_signature(X_train, model.predict(X_train))

# -------------------------------------------
# 2. Buat input example (WAJIB untuk tampil schema)
# -------------------------------------------
input_example = X_train[:2]   # 2 sample contoh input

# -------------------------------------------
# 3. Log model secara manual
# -------------------------------------------
with mlflow.start_run():
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("C", 100)
    mlflow.log_param("gamma", 0.1)

    mlflow.log_metric("test_score", model.score(X_test, y_test))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
