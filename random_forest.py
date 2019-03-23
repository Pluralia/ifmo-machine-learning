import decision_tree as dt
from sklearn.model_selection import train_test_split
import numpy as np


class RandomForest:
    def __init__(self, tree_num, max_depth):
        self.tree_num = tree_num
        self.max_depth = max_depth
        self.trees = [dt.DecisionTree(dt.gini_index, max_depth) for _ in range(tree_num)]
        self.tag_idx = []

    def build(self, data, labels):
        for i, tree in enumerate(self.trees):
            print(i + 1)
            train_data, _, train_labels, _ = train_test_split(data, labels, train_size=0.8, test_size=0.2)
            tags = np.random.permutation(data.shape[1])[:int(data.shape[1] * 0.8)]
            self.tag_idx.append(tags)
            tree.build(train_data[:, tags], train_labels)
        return self

    def predict(self, tree_num, max_depth, data):
        probability = self.predict_probability(tree_num, max_depth, data)
        return (probability >= 0.5).astype(np.int)

    def predict_probability(self, tree_num, max_depth, data):
        if tree_num > self.tree_num or max_depth > self.max_depth:
            print("Error number of trees")
            return None
        res = np.zeros(data.shape[0])
        for tree, tags in zip(self.trees[:tree_num], self.tag_idx[:tree_num]):
            res += tree.predict_probability(max_depth, data[:, tags])
        return res / tree_num
