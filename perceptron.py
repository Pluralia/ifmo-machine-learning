import numpy as np


class Perceptron:
    def __init__(self):
        self.pocket = None
        self.best_res = 0
        self.W = None
        self.X = None

    def build(self, data, labels, num):
        self.W = np.random.rand(data.shape[1] + 1).transpose()
        self.pocket = self.W
        self.X = np.hstack((np.ones((data.shape[0], 1)), data))
        self.best_res = np.sum(labels == self.predict_X(self.X))
        for n in range(num):
            for i, x in enumerate(self.X):
                y = self.predict_X(self.X)[i]
                if y != labels[i]:
                    if y == -1:
                        self.W += self.X[i]
                    elif y == 1:
                        self.W -= self.X[i]
                    curr_res = np.sum(labels == self.predict_X(self.X))
                    if self.best_res < curr_res:
                        self.best_res = curr_res
                        self.pocket = self.W
        return self.pocket

    def predict_X(self, data):
        return np.sign(self.W.dot(data.transpose()))

    def predict(self, data):
        X = np.hstack((np.ones((data.shape[0], 1)), data))
        return self.predict_X(X)

    def get_weights(self):
        return self.W