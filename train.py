import numpy as np
import pandas as pd
from sys import argv
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from config import MY_FEATURES, saving_file, saving_folder


class LogReg:

    def __init__(self, filename):

        self.costs = None
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

        self.thetas = []
        self.bs = []


    def convert_housenames(self):
        self.Y = np.asarray([np.float128(np.where(self.house_names == x)[0][0] + 1) for x in self.Y]).reshape(-1, 1)


        self.Y_1 = (self.Y.astype(int) == 1).astype(int).reshape(-1,1)
        self.Y_2 = (self.Y.astype(int) == 2).astype(int).reshape(-1,1)
        self.Y_3 = (self.Y.astype(int) == 3).astype(int).reshape(-1,1)
        self.Y_4 = (self.Y.astype(int) == 4).astype(int).reshape(-1,1)

        self.y_s = [self.Y_1, self.Y_2, self.Y_3, self.Y_4]

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

        g = (np.dot(theta.T,X) + b).T
        h = self.sigmoid(g)

        cost = (1/m) * np.sum(np.dot(-Y.T , np.log(h)) - 
        np.dot((1 - Y).T , np.log(1 - h)))

        dtheta = (1 / m) * np.dot(X, (h - Y))
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
            db = grads["db"]

            theta = theta - lr * dtheta
            b = b - lr * db
            costs.append(cost)
        params = {"theta": theta,
                  "b": b}
        grads = {"dtheta": dtheta,
                 "db": db}
        return params, grads, costs

    def save_coefficients(self, filename=saving_file, folder=saving_folder):
        results = []
        for theta, b in zip(self.thetas, self.bs):
          results.append({'b': str(b), 'theta': [str(th[0]) for th in theta.tolist()]})

        with open(os.join(folder, filename), 'w') as file:
            file.write(json.dumps(results))

    def model(self, X, Y, lr=0.001, num_iterations=10000, visu=0):
        theta, b = self.init_with_zeroes(X.shape[0])
        parametrs = []
        grads = []
        costs = []
        for i,y in enumerate(self.y_s):
          parameter, grad, cost = self.optimize(theta, b, X, y, num_iterations, lr)
          parametrs.append(parameter)

          grads.append(grad)
          costs.append(cost)

          self.thetas.append(parameter['theta'])
          self.bs.append(parameter['b'])
          if visu:
              self.show_costs(costs)

        self.save_coefficients(i)
        self.costs = costs
      
    def show_costs(self, costs):
          plt.plot([i for i in range(len(cost))],cost)
          plt.xlabel('iterations')
          plt.ylabel('cost')


if name == '__main__':
    if len(argv) != 2 and len(argv) != 3:
        print('Usage: python3 train.py {training_filename}')
        print('or python3 train.py {training_filename} -v')
        exit()
    try:
        filename = open(argv[1], w)
    except:
        print('No file')
        exit()
    if argv[2] == '-v':
        visu = 1
    log_reg = LogReg(argv[1])
    log_reg.model(log_reg.df, log_reg.Y, visu=visu)
