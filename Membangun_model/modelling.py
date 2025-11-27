
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

import os
import mlflow as mlf
import math

my_path = '/content/aapl.us.txt_preprocessing.csv'
df = pd.read_csv(my_path)
df.head()

data = df.filter(['Close_norm'])

dataset = data.values

training_data_len = math.ceil(len(dataset) * .8)

train_data = dataset[0:training_data_len, :]

X_train = []
y_train = []

#windowing
def split_data(data,x,y):
  for i in range(60, len(data)):
    x.append(data[i-60:i,0])
    y.append(data[i,0])

split_data(train_data, X_train,y_train)

X_train, y_train = np.array(X_train), np.array(y_train)

mlf.set_tracking_uri(uri="http://127.0.0.1:5000/")

test_data = dataset[training_data_len - 60:, :]

X_test = []
y_test = dataset[training_data_len:, :]

split_data(test_data,X_test,[])

with mlf.start_run():
  model = SVC()

  mlf.autolog()

  mlf.sklearn.log_model(
      sk_model=model,
      artifact_path="model",
  )
  model.fit(X_train, y_train)
  accuracy = model.score(X_test,y_test)