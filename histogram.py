from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 histogram.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.groupby(['Hogwarts House'])
    grouped_dfs = [group for group in df]
    colors = ['red', 'green', 'blue', 'yellow']
    names = []
    for i in range(len(grouped_dfs)):
        names.append(grouped_dfs[i][0])
        tmp_df = grouped_dfs[i][1]
        tmp_df = tmp_df.select_dtypes([np.number])
        tmp_df = tmp_df.drop(columns=['Index'])
        for column in tmp_df:
            plt.hist(tmp_df[column], color=colors[i], alpha=0.5, lw=3, label=names[i])
    plt.xlabel('Marks')
    plt.ylabel('Number of students')
    c1, c2, c3, c4 = create_legend()
    plt.legend([c1, c2, c3, c4], names)
    plt.show()
