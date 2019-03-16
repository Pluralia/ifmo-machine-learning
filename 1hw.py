import pandas as pd
from roc import *


# cancer
c_df = pd.read_csv("cancer.csv")
c_labels = np.array([1 if l == 'M' else 0 for l in c_df['label'].values])
c_df = c_df.values
c_data = c_df[:, 1:]

evaluate_best_features(c_data, c_labels)


# spam
s_df = pd.read_csv("spam.csv")
s_df = s_df.values
s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]

evaluate_best_features(s_data, s_labels)

print("OK")
