import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import roc
from naive_bayes import *

from sklearn.feature_extraction.text import CountVectorizer


# smsspam
df = pd.read_csv("smsspam.csv")
labels = (df["label"].values == 'spam') + 0
data = (CountVectorizer().fit_transform(np.array([x[0] for x in df.drop(["label"], axis=1).values])) != 0) * 1
data = data.toarray()

tr_data, val_data, tr_labels, val_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)


new_tr_labels, _ = mult_predict(tr_data, tr_labels, tr_data)
tr_accuracy = np.sum(new_tr_labels == tr_labels) / tr_data.shape[0]
new_val_labels, features_prob = mult_predict(tr_data, tr_labels, val_data)
val_accuracy = np.sum(new_val_labels == val_labels) / val_data.shape[0]
print("Result for train data: ", tr_accuracy)
print("Result for validate data: ", val_accuracy)

roc.evaluate_roc_auc(features_prob, val_labels, "accuracy: %.3f" % val_accuracy)
plt.show()
