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


def my_count(x):
    count = len(x)
    for elem in x:
        if math.isnan(elem):
            count -= 1
    return count

if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 computor.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.select_dtypes([np.number])
    counts = [my_count(df[column]) for column in df]
    means = [my_mean(df[column]) for column in df]
    stds = [my_std(df[column]) for column in df]
    mins = [my_mean(df[column]) for column in df]
    pers_25 = [my_percentile(df[column], 0.25) for column in df]
    pers_50 = [my_percentile(df[column], 0.50) for column in df]
    pers_75 = [my_percentile(df[column], 0.75) for column in df]
    maxes = [my_max(df[column]) for column in df]
    column_names = [column for column in df]
    all_raws = [counts, means, stds, mins, pers_25, pers_50, pers_75, maxes]
    new_df = pd.DataFrame(all_raws, columns=column_names, index=('count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'))
    print(new_df)
