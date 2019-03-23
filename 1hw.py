import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random_forest as rf
import roc


tree_num = 20
max_depth = 3


# cancer
# c_df = pd.read_csv("cancer.csv")
# c_labels = np.array([1 if l == 'M' else 0 for l in c_df['label'].values])
# c_df = c_df.values
# c_data = c_df[:, 1:]
# c_tr_data, c_te_data, c_tr_labels, c_te_labels = train_test_split(c_data, c_labels, train_size=0.8, test_size=0.2)
#
# forest = rf.RandomForest(tree_num, max_depth).build(c_tr_data, c_tr_labels)
#
# new_labels = forest.predict(tree_num, max_depth, c_te_data)
# accuracy = np.mean(new_labels == c_te_labels)
#
# new_tag = forest.predict_probability(tree_num, max_depth, c_te_data)
# roc.evaluate_roc_auc(new_tag, c_te_labels, "accuracy: %.3f" % accuracy)


#spam
s_df = pd.read_csv("spam.csv")
s_df = s_df.values
s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]
s_tr_data, s_te_data, s_tr_labels, s_te_labels = train_test_split(s_data, s_labels, train_size=0.8, test_size=0.2)

forest = rf.RandomForest(tree_num, max_depth).build(s_tr_data, s_tr_labels)

new_labels = forest.predict(tree_num, max_depth, s_te_data)
accuracy = np.mean(new_labels == s_te_labels)

new_tag = forest.predict_probability(tree_num, max_depth, s_te_data)
roc.evaluate_roc_auc(new_tag, s_te_labels, "accuracy: %.3f" % accuracy)


plt.show()
print("OK")