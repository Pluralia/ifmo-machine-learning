import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import decision_tree as dt


def best_depth(impurity, data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)
    print(np.mean(train_labels), np.mean(test_labels))

    for max_depth in range(1, 11):
        tree = dt.DecisionTree(impurity, max_depth).build(train_data, train_labels)
        new_labels = tree.predict(test_data)
        accuracy = np.mean(new_labels == test_labels)
        print("max_depth: ", max_depth, " | ", "accuracy: ", accuracy)


# cancer
c_df = pd.read_csv("cancer.csv")
c_labels = np.array([1 if l == 'M' else 0 for l in c_df['label'].values])
c_df = c_df.values
c_data = c_df[:, 1:]

print("cancer.csv")
print("Misclassification error")
best_depth(dt.misclass_err, c_data, c_labels)
print("Entropy")
best_depth(dt.entropy, c_data, c_labels)
print("Gini index")
best_depth(dt.gini_index, c_data, c_labels)


# spam
s_df = pd.read_csv("spam.csv")
s_df = s_df.values
s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]

print("spam.csv")
print("Misclassification error")
best_depth(dt.misclass_err, s_data, s_labels)
print("Entropy")
best_depth(dt.entropy, s_data, s_labels)
print("Gini index")
best_depth(dt.gini_index, s_data, s_labels)


print("OK")
