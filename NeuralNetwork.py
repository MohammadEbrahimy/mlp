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
            self.weights[i][-1, 1:] = self.weights[i][-1, 0]
    # Activation function
    def sigmoid(self, X):
        return 1 / (1+np.exp(-X))

    # Step function for output layer
    def step_function(self, output):
        if output >= 0:
            return 1
        else:
            return 0
    
    # Normalizing Data
    def normalize(self, x):
        return (x-min(x))/(max(x)-min(x))

    # FeedForward
    def forward(self, X):
        X = np.append(X, 1)
        h = [X]
        for i in range(self.layer_number):
            X = self.weights[i].T @ X
            X = self.sigmoid(X)
            X = np.append(X, 1)
            h.append(X)
        return X[0], h

    #BackPropagation
    def back_propagation(self, y, y_hat, h, learning_rate):
        error_out = abs(y_hat*(1-y_hat)*(y-y_hat))
        # print('y is :', y)
        # print('y_hat is :', y_hat)
        # print("error is :", error_out)
        end_layer = True
        Dw = []
        dh = []
        Sum = []
        dh.append(np.array([error_out]))
        for i in reversed(range(len(self.weights)+1)):
            if end_layer:
                my_sum = self.weights[i-1][:-1] * error_out
                Sum.append(my_sum)
                dh.append(np.array([(h[i][:-1]*(1-h[i][:-1]))]).T * my_sum)
                # dw = ([number * dh[0] for number in h[i-1]])
                # Dw.append(np.array([[number * learning_rate for number in dw]]))
                end_layer = False
            else:
                if i != 0:
                    w = self.weights[i-1][:-1]
                    for e in range(w.shape[1]): 
                        temp_sum = w @ dh[-1]
                        for n in  reversed(range(len(dh)-1)):
                            for j in range(len(dh[n])):
                                temp_sum[j][0] += dh[n][j]
                    if i != 1:
                        dh_temp = np.array([(h[i-1][:-1]*(1-h[i-1][:-1]))]).T * temp_sum
                        dh.append(np.array([dh_temp.sum(axis=1, dtype='float')]).T)
                    dw = (h[i] * dh[abs(i-(len(self.weights)-1))]).T
                    # print("-----------------------------------------------")
                    Dw.append(np.array([[number * learning_rate for number in dw]]))
                else:
                    dw = (h[i] * dh[abs(i-(len(self.weights)-1))]).T
                    Dw.append(np.array([[number * learning_rate for number in dw]]))
        # print(Dw[0])
        # print(Dw[1])
        # print(Dw[2])
        # print(Dw[3])

    def predict(self, x, y, learning_rate):
        self.layers["Layer0"] = x.shape[1]
        self.initialize_parameters(self.layers)
        for i in range(y.shape[0]):
            x_normal = self.normalize(x[i, :])
            y_hat, h = self.forward(x_normal)
            self.back_propagation(y[i], y_hat, h, learning_rate)
