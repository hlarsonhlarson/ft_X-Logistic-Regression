from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 histogram.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1]).dropna()
    column_names = [column for column in df.select_dtypes([np.number]).drop(columns=['Index'])]
    for i in range(len(column_names)):
        if len(column_names) - i == 2:
            break
        fig, axes = plt.subplots(len(column_names) - i - 1, figsize=(30, 30))
        for j in range(i, len(column_names) - 1):
            axes[j - i].scatter(df[column_names[i]], df[column_names[j + 1]])
            axes[j - i].legend([f'{column_names[i]}-{column_names[j + 1]}'])
        plt.show()
    plt.scatter(df[column_names[-2]], df[column_names[-1]])
    plt.legend([f'{column_names[-2]}-{column_names[-1]}'])
    plt.show()
