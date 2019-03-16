import collections
import numpy as np


def misclass_err(labels):
    _, counts = np.unique(labels, return_counts=True)
    return 1 - counts.max() / counts.sum()


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -np.sum(norm_counts * np.log(norm_counts))


def gini_index(labels):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return np.sum(norm_counts * (1 - norm_counts))


class DecisionTree:
    def __init__(self, impurity, max_depth):
        self.impurity = impurity
        self.max_depth = max_depth
        self.depth = 0
        self.is_leaf = True
        self.node = None
        self.threshold = None
        self.tag_id = None
        self.left = None
        self.right = None

    def get_gain(self, labels):
        return self.impurity(labels) * len(labels)

    def build(self, data, labels):
        self.node = collections.Counter(labels).most_common(1)[0][0]
        if self.max_depth == 0:
            return self
        res_IG = 0
        for tag_id in range(data.shape[1]):
            tag = data[:, tag_id]
            for threshold in tag:
                l_labels = labels[tag <= threshold]
                r_labels = labels[tag > threshold]
                if not (len(l_labels) and len(r_labels)):
                    continue
                IG = self.get_gain(labels) - self.get_gain(l_labels) - self.get_gain(r_labels)
                if res_IG < IG:
                    res_IG = IG
                    self.threshold = threshold
                    self.tag_id = tag_id
        if res_IG > 0:
            self.create_branches(res_IG, data, labels)
        return self

    def create_branches(self, IG, data, labels):
        self.is_leaf = False
        l_data = data[:, self.tag_id] <= self.threshold
        self.left = DecisionTree(self.impurity, self.max_depth - 1).build(data[l_data], labels[l_data])
        r_data = data[:, self.tag_id] > self.threshold
        self.right = DecisionTree(self.impurity, self.max_depth - 1).build(data[r_data], labels[r_data])
        self.depth = max(self.left.depth, self.right.depth) + 1

    def predict(self, data):
        if self.is_leaf:
            return np.zeros(data.shape[0]) + self.node
        labels = np.zeros(data.shape[0])
        l_data = data[:, self.tag_id] <= self.threshold
        labels[l_data] = self.left.predict(data[l_data])
        r_data = data[:, self.tag_id] > self.threshold
        labels[r_data] = self.right.predict(data[r_data])
        return labels
