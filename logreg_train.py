import numpy as np
import pandas as pd


MY_FEATURES = ["Herbology", "Ancient Runes", "Astronomy", "Charms", "Defense Against the Dark Arts"]

class LogReg:

    def __init__(self, filename):
        self.mean = None
        self.std = None
        self.df = pd.open_csv(filename)
        self.Y = self.df['Hogwarts House']
        self.df = self.df[MY_FEATURES]
        self.df = normalize_data(self.df)
        self.theta = None
        self.b = None

    def normalize_data(data):
        self.mean = data.mean()
        self.std = data.std()
        return (data - data.mean()) / data.std()

    def init_with_zeroes(self, dim):
        self.theta = np.zeros(shape=(dim, 1))
        self.b = 0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def propagate(self, theta, b, X, Y):
        m = X.shape[1]
        g = np.dot(theta.T,X) + b
        h = self.sigmoid(g)
        cost = (-1/m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)))
        dtheta = (1 / m) * np.dot(X, (h - Y).T)
        db = (1 / m) * np.sum(h - Y)

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
        params = {"theta": theta
                  "b": b}
        grads = {"dtheta": dtheta,
                 "db": db}
        return params, grads, cost

    def save_coefficients(self, filename='coefficients.json', folder='data'):
        b = b - theta * self.mean/self.std
        theta = theta / self.std
        result = {'b': b, 'theta': theta, 'mean': self.mean, 'std': self.std}
        with open(os.path.join(folder, filename), 'w') as file:
            file.write(json.dumps(result))

    def model(self, X, Y, num_iterations=100000, lr, visu=0):
        theta, b = self.init_with_zeroes(X.shape[0])
        parameters, grads, costs = self.optimize(theta, b, X, Y, num_iterations, lr)
        self.save_coefficients(parameters['theta'], paramets['b'])

