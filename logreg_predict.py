import os
import json
from sys import argv
import numpy as np
import pandas as pd
from config import MY_FEATURES, saving_file, saving_folder, houses_filename, ALL_NEEDED


def error(msg):
  print(msg)
  exit()

def get_coefficients(coefficients_filename, folder):
    full_filename = saving_file
    if os.path.exists(full_filename) and os.path.isfile(full_filename):
        with open(full_filename, 'r') as file:
            file = file.read()
        try:
            data = json.loads(file)
            return data
        except:
            error(f'format of the file "{full_filename}" is incorrect (this must be json contains dict with 2 keys: "teta_0" and "teta_1", all them is floats or ints)')
    error(f'the ("{full_filename}") file with coefficients is not exists')

def sigmoid(X):
  return 1 / (1 + np.exp(-X))

def get_housenames(coeff_dict, X):
  coeff_dict['theta'] = np.array(coeff_dict['theta'], dtype=float)
  coeff_dict['b'] = np.float(coeff_dict['b'])
  return sigmoid(coeff_dict['b'] + np.dot(coeff_dict['theta'], X))

def predict(data_X):
    coeffs = get_coefficients(saving_file, saving_folder)
    our_label = []
    for index, row in data_X.iterrows():
        X = row.to_numpy(dtype=float)
        values = []
        for i in range(len(coeffs)):
            values.append(get_housenames(coeffs[i], X))
        my_values = np.argmax(values) + 1
        our_label.append(my_values)
    return our_label

def name_houses(prediction_array):
    full_filename = houses_filename
    if os.path.exists(full_filename) and os.path.isfile(full_filename):
        with open(full_filename, 'r') as file:
            file = file.read()
        try:
            houses = json.loads(file)
        except:
            error(f'format of the file "{full_filename}" is incorrect (this must be json contains dict with 2 keys: "teta_0" and "teta_1", all them is floats or ints)')
    else:
        error(f'the ("{full_filename}") file with coefficients is not exists')
    ans = [houses[x - 1] for x in prediction_array]

    return ans


if __name__ == '__main__':
    if len(argv) != 2:
        print('Usage: python3 predict.py {filename}')
        exit()
    try:
        data = pd.read_csv(argv[1])
        data = data[ALL_NEEDED]
    except:
        print(f'Wrong file or file format message is {e}')
        exit()
    data_X = data[MY_FEATURES]
    data_X = data_X.fillna(data_X.mean())
    data_X = (data_X - data_X.mean()) / data_X.std()
    prediction_array = predict(data_X)
    house_names = name_houses(prediction_array)
    df = pd.DataFrame({'Index': [x for x in range(len(house_names))], 'Hogwarts House': [x for x in house_names]})
    df.to_csv('houses.csv', index=False)
