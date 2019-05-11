import random
import pandas as pd
import matplotlib.pyplot as plt


def plot_l1(x1, y1, x2, y2):
    dist = 0
    if x1 != x2:
        plt.arrow(x1, y1, x2 - x1, 0, color='r', shape='full', length_includes_head=True, head_width=15.)
        x1 = x2
        dist += abs(x2 - x1)
    if y1 != y2:
        plt.arrow(x1, y1, 0, y2 - y1, color='r', shape='full', length_includes_head=True, head_width=15.)
        dist += abs(y2 - y1)
    return dist


def get_data():
    # tsp
    df = pd.read_csv("tsp.csv")
    df = df.values
    x = df[:, 1]
    y = df[:, 2]
    size = x.shape[0]
    return size, x, y


def build_plot(idxs, x, y):
    dist = 0
    plt.figure()
    for i in range(1, len(idxs)):
        fr = idxs[i - 1]
        to = idxs[i]
        dist += plot_l1(x[fr], y[fr], x[to], y[to])
    plt.scatter(x, y, c='black')

    plt.title(f'Distance length: {dist}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
