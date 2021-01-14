from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    if len(argv) != 2:
        print('Incorrect input. Usage: python3 histogram.py "{your expression}"')
        exit(1)
    df = pd.read_csv(argv[1])
    df = df.drop(columns=['Index'])
    if not df.loc[:,'Hogwarts House'].isnull().values.any():
        sns.pairplot(df.dropna(), hue="Hogwarts House")
    else:
        df = df.drop(columns=['Hogwarts House'])
        sns.pairplot(df.dropna())
    plt.tight_layout()
    plt.savefig('features.pdf')

