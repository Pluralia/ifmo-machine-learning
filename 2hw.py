import copy
import pandas as pd
from functools import partial
import lib
import random


def random_swap(size, path):
    for i in range(1000):
        pair_num = 20
        tmp_path = list(range(size))
        random.shuffle(tmp_path)
        idx_x = tmp_path[:pair_num]
        random.shuffle(tmp_path)
        idx_y = tmp_path[:pair_num]
        best_dist = 1e6
        best_path = None
        for (i, j) in list(zip(idx_x, idx_y)):
            swap_path = copy.copy(path)
            swap_path[i], swap_path[j] = swap_path[j], swap_path[i]
            swap_dist = count_dist(swap_path)
            if best_dist > swap_dist:
                best_dist = swap_dist
                best_path = swap_path
        path = best_path
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
    path = random_swap(size, init_path)
    dist = count_dist(path)
    if dist < best_dist:
        best_path = path
        best_dist = dist
    print(i, best_dist)

lib.build_plot(x, y, best_path)
