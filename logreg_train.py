import numpy as np
import pandas as pd
from sys import argv
import os
import json


MY_FEATURES = ["Herbology", "Ancient Runes", "Astronomy", "Charms", "Defense Against the Dark Arts"]

class LogReg:

    def __init__(self, filename):
        self.mean = None
        self.std = None
        self.df = pd.read_csv(filename)
        self.df = self.df.dropna()
        self.Y = self.df['Hogwarts House']
        self.house_names = self.Y.unique()
        self.df = self.df[MY_FEATURES]
        self.df = self.normalize_data(self.df)

        self.df = self.df.to_numpy(dtype=np.float128)

        self.convert_housenames()

        #flattern
        self.df = self.df.reshape(self.df.shape[0], -1).T
        self.Y = self.Y.reshape(self.Y.shape[0], -1).T

        self.theta = None
        self.b = None

    def convert_housenames(self):
        self.Y = [np.float128(np.where(self.house_names == x)[0][0] + 1) for x in self.Y]
        self.Y = np.array(self.Y)

    def normalize_data(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return (data - data.mean()) / data.std()

    def init_with_zeroes(self, dim):
        self.theta = np.zeros(shape=(dim, 1))
        self.b = 0
        return self.theta, self.b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def propagate(self, theta, b, X, Y):
        m = X.shape[1]
        g = np.dot(theta.T,X) + b
        h = self.sigmoid(g)
        cost = (-1/m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
        dtheta = (1 / m) * np.dot(X, (h - Y).T)
        db = (1 / m) * np.sum(h - Y)

        cost = np.squeeze(cost)

        grads = {"dtheta": dtheta,
                 "db": db}
        return grads, cost

    def optimize(self, theta, b, X, Y, num_iterations, lr):
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(theta, b, X, Y)

            dtheta = grads["dtheta"]
            db= grads["db"]

            theta = theta - lr * dtheta
            b = b - lr * db

            if i % 100 == 0:
                costs.append(cost)
        params = {"theta": theta,
                  "b": b}
        grads = {"dtheta": dtheta,
                 "db": db}
        return params, grads, cost

    def save_coefficients(self, filename='coefficients.json', folder='data'):
        print(self.theta)
        self.theta = self.theta / np.mean(self.std)
        self.b = self.b - self.theta * np.mean(self.mean)/np.mean(self.std)
        result = {'b': self.b.tolist(), 'theta': self.theta.tolist()}
        with open(os.path.join(filename), 'w+') as file:
            file.write(json.dumps(result))

    def model(self, X, Y, lr, num_iterations=100000, visu=0):
        theta, b = self.init_with_zeroes(X.shape[0])
        parameters, grads, costs = self.optimize(theta, b, X, Y, num_iterations, lr)
        print(self.theta)
        #self.save_coefficients()

if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 logreg.py "{your expression}"')
        exit(1)
    log_reg = LogReg(argv[1])
    log_reg.model(log_reg.df, log_reg.Y, 0.01)
