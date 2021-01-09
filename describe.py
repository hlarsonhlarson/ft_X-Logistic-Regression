import numpy as np
import pandas as pd
from sys import argv
import math


def my_mean(x):
    summary = 0
    length = len(x)
    for i in range(len(x)):
        if not math.isnan(x[i]):
            summary += x[i]
        else:
            length -= 1
    return summary / length


def my_std(x):
    m = my_mean(x)
    ans = 0
    length = len(x)
    for elem in x:
        if math.isnan(elem):
            length -= 1
        else:
            ans += (elem - m)**2
    ans /= length
    return np.sqrt(ans)


def my_min(x):
    ans = x[0]
    for elem in x:
        if elem < ans:
            ans = elem
    return ans


def my_max(x):
    ans = x[0]
    for elem in x:
        if elem > ans:
            ans = elem
    return ans


if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 computor.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.select_dtypes([np.number])
    print('my', my_std(df['Potions']), '\n', 'their', np.std(df['Potions']))
    '''
    for column in df:
        print(column)
        print('my', my_mean(df[column]), '\n', 'their', np.mean(df[column]))
    '''
