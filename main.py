import numpy as np
import pandas as pd
import NeuralNetwork as NN

# Importing tratin and test data
train = pd.read_csv("dataset/nba_logreg.csv", header=None)
test = pd.read_csv("dataset/test.csv", header=None)

# Split the train and test data
x_train = train.values[1:, 1:20].astype(np.float)
y_train = train.values[1:, 20].astype(np.float)
x_test = test.values[1:, 2:21].astype(np.float)
y_test = test.values[1:, 21].astype(np.float)

test = NN.MlpClassifier()
test.add_layer(5)
test.add_layer(23)
test.add_layer(5)
a = test.forward(x_train[1, :])
print(a)
