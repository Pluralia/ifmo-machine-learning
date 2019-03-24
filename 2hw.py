import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random_forest as rf
import roc


tree_num = [5, 10, 20, 30, 50, 100]
max_depth = [2, 3, 5, 7, 10]


# cancer
# c_df = pd.read_csv("cancer.csv")
# c_labels = np.array([1 if l == 'M' else 0 for l in c_df['label'].values])
# c_df = c_df.values
# c_data = c_df[:, 1:]
# c_tr_data, c_te_data, c_tr_labels, c_te_labels = train_test_split(c_data, c_labels, train_size=0.8, test_size=0.2)
#
# forest = rf.RandomForest(100, 10).build(c_tr_data, c_tr_labels)
#
# best_acc = 0
# best_tn = 0
# best_md = 0
# for tn in tree_num:
#     for md in max_depth:
#         new_labels = forest.predict(tn, md, c_te_data)
#         accuracy = np.mean(new_labels == c_te_labels)
#         print("tree_num: ", tn, " | ", "max_depth: ", md, " | ", "accuracy: %.3f" % accuracy)
#         if accuracy > best_acc:
#             best_acc = accuracy
#             best_tn = tn
#             best_md = md
#
# new_tag = forest.predict_probability(best_tn, best_md, c_te_data)
# roc.evaluate_roc_auc(new_tag, c_te_labels, "tree_num: " + best_tn.__str__() + " | max_depth: " + best_md.__str__() + " | accuracy: %.3f" % best_acc)


#spam
s_df = pd.read_csv("spam.csv")
s_df = s_df.values
s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]
s_tr_data, s_te_data, s_tr_labels, s_te_labels = train_test_split(s_data, s_labels, train_size=0.8, test_size=0.2)

forest = rf.RandomForest(100, 10).build(s_tr_data, s_tr_labels)

best_acc = 0
best_tn = 0
best_md = 0
for tn in tree_num:
    for md in max_depth:
        new_labels = forest.predict(tn, md, s_te_data)
        accuracy = np.mean(new_labels == s_te_labels)
        print("tree_num: ", tn, " | ", "max_depth: ", md, " | ", "accuracy: %.3f" % accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            best_tn = tn
            best_md = md

new_tag = forest.predict_probability(best_tn, best_md, s_te_data)
roc.evaluate_roc_auc(new_tag, s_te_labels, "tree_num: " + best_tn.__str__() + " | max_depth: " + best_md.__str__() + " | accuracy: %.3f" % best_acc)


plt.show()
print("OK")