import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random_forest as rf


# cancer
# name = 'cancer'
# df = pd.read_csv("cancer.csv")
# labels = np.array([1 if l == 'M' else 0 for l in df['label'].values])
# df = df.values
# data = df[:, 1:]
# data = data / np.max(data, axis=0)

# spam
name = 'spam'
df = pd.read_csv("spam.csv")
df = df.values
labels = df[:, -1]
data = df[:, 0:-1]
data = data / np.max(data, axis=0)

tr_data, te_data, tr_labels, te_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

# Random Forest
rf_start_build = time.time()
tree_num, max_depth = 5, 3
forest = rf.RandomForest(tree_num, max_depth).build(tr_data, tr_labels)
rf_start_predict = time.time()
res = forest.predict(tree_num, max_depth, te_data)
rf_finish = time.time()
acc = round(sum(res == te_labels) / te_labels.shape[0], 3)
print("RandomForest:")
print("Accuracy: ", acc)
print("All time: ", rf_finish - rf_start_build)
print("Predict time: ", rf_finish - rf_start_predict)


# SVM
svm_start_build = time.time()
clf = SVC(gamma='scale', kernel='linear', degree=1, probability=True)
clf.fit(tr_data, tr_labels)
svm_start_predict = time.time()
res = clf.predict(te_data)
svm_finish = time.time()
acc = round(sum(res == te_labels) / te_labels.shape[0], 3)
print("SVM:")
print("Accuracy: ", acc)
print("All time: ", svm_finish - svm_start_build)
print("Predict time: ", svm_finish - svm_start_predict)