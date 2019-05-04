import pandas as pd
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import roc
import re
from naive_bayes import *


def unify_text(msg):
    msg = re.compile('\w+').findall(msg)
    # lower_case_words = [word.lower() for word in msg]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word.lower()) for word in msg]
    return stemmed_words


# spam
name = 'smsspam'
df = pd.read_csv("smsspam.csv")
df = df.values
labels = np.array([1 if l == 'ham' else 0 for l in df[:, 0]], int)
data = df[:, 1]

data = np.array([unify_text(msg) for msg in data])
word_dict = dict()
idx_dict = dict()
idx = 0
for j in range(0, data.shape[0]):
    for word in data[j]:
        if word_dict.get(word) is None:
            word_dict[word] = idx
            idx_dict[idx] = word
            idx += 1
for (k, v) in idx_dict.items():
    if word_dict[v] != k:
        print("Dictionaries do not match")

formatting_data = np.zeros((data.shape[0], len(word_dict)))
for j in range(0, data.shape[0]):
    for word in data[j]:
        formatting_data[j, word_dict[word]] += 1
data = formatting_data
tr_data, val_data, tr_labels, val_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)


new_tr_labels, _ = mult_predict(tr_data, tr_labels, tr_data)
tr_accuracy = np.sum(new_tr_labels == tr_labels) / tr_data.shape[0]
new_val_labels, features_prob = mult_predict(tr_data, tr_labels, val_data)
val_accuracy = np.sum(new_val_labels == val_labels) / val_data.shape[0]
print("Result for train data: ", tr_accuracy)
print("Result for validate data: ", val_accuracy)

roc.evaluate_roc_auc(features_prob, val_labels, "accuracy: %.3f" % val_accuracy)
plt.show()
