import copy
import pandas as pd
from functools import partial
import lib
import random


def climb(size, path):
    curr_dist = count_dist(path)
    while True:
        prev_dist = curr_dist
        best_dist = 1e6
        best_i, best_j = 0, 0
        for i in range(size):
            for j in range(i + 1, size):
                curr_path = copy.copy(path)
                curr_path[i], curr_path[j] = curr_path[j], curr_path[i]
                swap_dist = count_dist(curr_path)
                if swap_dist < best_dist:
                    best_dist = swap_dist
                    best_i, best_j = i, j

        path[best_i], path[best_j] = path[best_j], path[best_i]
        curr_dist = count_dist(path)
        if curr_dist >= prev_dist:
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
    path = climb(size, init_path)
    dist = count_dist(path)
    if dist < best_dist:
        best_path = path
        best_dist = dist
    print(i, best_dist)

lib.build_plot(x, y, best_path)
