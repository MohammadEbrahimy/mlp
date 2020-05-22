import numpy as np


class MlpClassifier(object):

    def __init__(self):
        self.weights = []
        self.layers = {"Layer0": 0}
        self.layer_number = 0

    # Add layer
    def add_layer(self, number_of_nodes):
        self.layer_number += 1
        self.layers["Layer"+str(self.layer_number)] = number_of_nodes

    # Initialize Weights and bias
    def initialize_parameters(self, layers):
        for i in range(self.layer_number):
            self.weights.append(np.random.rand(layers["Layer"+str(i)]+1,
                                               layers["Layer"+str(i+1)]))

    # Activation function
    def sigmoid(self, X):
        return 1 / (1+np.exp(-X))

    # FeedForward
    def forward(self, X):
        self.layers["Layer0"] = X.shape[0]
        X = np.append(X, 1)
        self.initialize_parameters(self.layers)
        for i in range(self.layer_number):
            X = self.weights[i].T @ X
            X = self.sigmoid(X)
            X = np.append(X, 1)
        return X
