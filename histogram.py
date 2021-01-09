from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 computor.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.groupby(['Hogwarts House'])
    grouped_dfs = [group for group in df]
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(len(grouped_dfs)):
        name = grouped_dfs[i][0]
        tmp_df = grouped_dfs[i][1]
        tmp_df = tmp_df.select_dtypes([np.number])
        print(tmp_df, type(tmp_df))
        tmp_df = tmp_df.drop(columns=['Index'])
        print(tmp_df, type(tmp_df))
        for column in tmp_df:
            plt.hist(column, color=colors[i])
        break

    plt.savefig('1.png')
