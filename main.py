import numpy as np
import pandas as pd
import NeuralNetwork as NN


# Importing tratin and test data
train = pd.read_csv("dataset/nba_logreg.csv", header=None)
test = pd.read_csv("dataset/test.csv", header=None)

train = train.dropna()
test = test.dropna()

# Split the train and test data
x_train = train.values[1:, 1:20].astype(np.float)
y_train = train.values[1:, 20].astype(np.float)
x_test = test.values[1:, 2:21].astype(np.float)
y_test = test.values[1:, 21].astype(np.float)

mlp = NN.MlpClassifier()
mlp.add_layer(10)
mlp.add_layer(15)
mlp.add_layer(5)
mlp.add_layer(2)
mlp.add_layer(1)

mlp.learn(x_train, y_train, 0.08, 500)
mlp.predict(x_test, y_test)

