import numpy as np


class Genetic:
    def __init__(self, count_dist, size):
        self.size = size
        self.count_dist = count_dist
        self.TOP_POPULATION_CUTTOF = 75
        self.POP_SIZE = 600
        self.NUM_CROSSOVERS = 55

        self.cur_population = []
        for i in range(self.POP_SIZE):
            cur = np.random.choice(range(self.size), self.size, replace=False)
            self.cur_population.append(cur)

    def crossover_and_mutate(self):
        for it in range(self.NUM_CROSSOVERS):
            idx1, idx2 = np.random.choice(range(self.size), 2, replace=False)
            c1 = self.crossover(self.cur_population[idx1], self.cur_population[idx2])
            c2 = self.crossover(self.cur_population[idx2], self.cur_population[idx1])
            self.cur_population.append(self.mutate(c1, 5))
            self.cur_population.append(self.mutate(c2, 5))

    def select(self):
        # num_unic_new = len(set(self.cur_population[self.POP_SIZE:self.POP_SIZE+self.NUM_CROSSOVERS]))
        # self.cur_population = self.cur_population[:self.POP_SIZE+num_unic_new]
        self.cur_population.sort(key=self.count_dist)
        self.cur_population = self.cur_population[:self.POP_SIZE]

    def crossover(self, x, y):
        result = np.zeros(self.size, dtype=np.int32) - 1

        for i in range(self.size // 2):
            result[i] = x[i]

        place_idx = self.size // 2
        for i in range(self.size):
            if all(result != y[i]):
                result[place_idx] = y[i]
                place_idx += 1
        return result

    def mutate(self, x, num_mut):
        for _ in range(num_mut):
            idx1, idx2 = np.random.choice(range(self.size), 2, replace=False)
            x[idx1], x[idx2] = x[idx2], x[idx1]
        return x
