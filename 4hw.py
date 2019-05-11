import copy
import math
import numpy as np
import pandas as pd
from functools import partial
import lib
import random


def annealing(size, path):
    temp = 1e6
    delta = np.zeros((size, size))
    while True:
        dist = count_dist(path)
        for i in range(size):
            for j in range(i + 1, size):
                swap_path = copy.copy(path)
                swap_path[i], swap_path[j] = swap_path[j], swap_path[i]
                swap_dist = count_dist(swap_path)
                delta[i][j] = dist - swap_dist
        probs = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                probs[i][j] = math.exp(delta[i][j] / temp)
        probs /= np.sum(probs)
        # prob = np.max(probs)
        # if np.random.uniform(0.0, 1.0) < prob:
        (i, j) = np.unravel_index(probs.argmax(), probs.shape)
        path[i], path[j] = path[j], path[i]
        temp *= 0.5
        if temp <= 1:
            return path


# tsp
df = pd.read_csv("tsp.csv")
df = df.values
x = df[:, 1]
y = df[:, 2]
size = x.shape[0]
count_dist = partial(lib.count_dist, x, y)

best_path = list(range(size))
best_dist = 1e6

for i in range(100):
    init_path = list(range(size))
    random.shuffle(init_path)
    path = annealing(size, init_path)
    dist = count_dist(path)
    if dist < best_dist:
        best_path = path
        best_dist = dist
    print(i, best_dist)
    if i % 10 == 0:
        lib.build_plot(x, y, best_path)

