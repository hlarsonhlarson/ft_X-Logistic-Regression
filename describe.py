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

def my_percentile(x, percent, key=lambda x:x):
    x = sorted(x)
    x = [x for x in x if not math.isnan(x)]
    x = sorted(x)
    needed_length = (len(x) - 1)* percent
    floor = math.floor(needed_length)
    ceil = math.ceil(needed_length)
    if floor == ceil:
        return key(x[int(needed_length)])
    d0 = key(x[int(floor)]) * (ceil - needed_length)
    d1 = key(x[int(ceil)]) * (needed_length - floor)
    return d0 + d1


if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 computor.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.select_dtypes([np.number])
    df.dropna()
    print('my', my_percentile(df['Potions'], 0.25), '\n', 'their', np.nanpercentile(df['Potions'], 25))
    '''
    for column in df:
        print(column)
        print('my', my_mean(df[column]), '\n', 'their', np.mean(df[column]))
    '''
