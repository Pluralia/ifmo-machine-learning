import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("blobs2.csv").values
labels = df[:, -1]
data = df[:, 0:-1]
print(labels)
print(data)

# s_tr_data, s_te_data, s_tr_labels, s_te_labels = train_test_split(s_data, s_labels, train_size=0.8, test_size=0.2)


plt.show()
print("OK")