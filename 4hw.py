import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import decision_tree as dt
import roc


def get_best_tree(impurity, data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

    max_max_deps = 10
    accuracy_list = np.empty(max_max_deps)
    res_tree = None
    res_accuracy = 0
    for max_depth in range(1, max_max_deps + 1):
        print(max_depth)
        tree = dt.DecisionTree(impurity, max_depth).build(train_data, train_labels)
        new_labels = tree.predict(test_data)
        accuracy = np.mean(new_labels == test_labels)
        if res_accuracy < accuracy:
            res_tree = tree
            res_accuracy = accuracy
        accuracy_list[max_depth - 1] = accuracy

    plt.figure()
    plt.plot(np.arange(max_max_deps) + 1, accuracy_list)
    plt.title(impurity.__name__)
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.pause(0.001)

    return res_tree


plt.ion()

# cancer
# c_df = pd.read_csv("cancer.csv")
# c_labels = np.array([1 if l == 'M' else 0 for l in c_df['label'].values])
# c_df = c_df.values
# c_data = c_df[:, 1:]
#
# for i, impurity in enumerate([dt.misclass_err, dt.entropy, dt.gini_index]):
#     print("impurity: " + impurity.__name__)
#     tree = get_best_tree(impurity, c_data, c_labels)
#     roc.evaluate_best_features(c_data, tree.predict(c_data))


# spam
s_df = pd.read_csv("spam.csv")
s_df = s_df.values
s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]

for i, impurity in enumerate([dt.misclass_err, dt.entropy, dt.gini_index]):
    tree = get_best_tree(impurity, s_data, s_labels)
    roc.evaluate_best_features(s_data, tree.predict(s_data))

plt.ioff()
plt.show()

print("OK")
