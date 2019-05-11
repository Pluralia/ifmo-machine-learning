import pandas as pd
from functools import partial
import lib
from genetic import Genetic


# tsp
df = pd.read_csv("tsp.csv")
df = df.values
x = df[:, 1]
y = df[:, 2]
size = x.shape[0]
count_dist = partial(lib.count_dist, x, y)

gen = Genetic(count_dist, size)

for i in range(1000):
    gen.crossover_and_mutate()
    gen.select()
    best_path = gen.cur_population[0]
    print(i, count_dist(best_path))

lib.build_plot(x, y, best_path)
