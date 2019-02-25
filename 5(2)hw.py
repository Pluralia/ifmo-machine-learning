import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from kNN import getLOO


# 5
maxK = 10
scaler = MinMaxScaler()


c_df = pd.read_csv('cancer.csv')
c_df = c_df.values

c_labels = c_df[:, 0]
c_data = c_df[:, 1:]
scaler.fit(c_data)
c_data = scaler.transform(c_data)

c_labels_dict = dict(zip(np.unique(c_labels), np.arange(c_labels.shape[0])))
print(c_labels_dict)
c_labels = np.array(list(map(lambda x: c_labels_dict[x], c_labels)))

c_loo_res = getLOO(maxK, c_data, c_labels)
print(c_loo_res)


s_df = pd.read_csv('spam.csv')
s_df = s_df.values

s_labels = s_df[:, -1]
s_data = s_df[:, 0:-1]
scaler.fit(s_data)
s_data = scaler.transform(s_data)

s_labels_dict = dict(zip(np.unique(s_labels), np.arange(s_labels.shape[0])))
print(s_labels_dict)
s_labels = np.array(list(map(lambda x: s_labels_dict[x], s_labels)))

s_loo_res = getLOO(maxK, s_data, s_labels)
print(s_loo_res)

plt.subplot(211)
plt.plot(np.arange(0, maxK) + 1, c_loo_res, 'b-')
plt.axis([1, maxK, np.min(c_loo_res), np.max(c_loo_res)])
plt.xlabel('Number of neighbors')
plt.ylabel('LOO-metrics')
plt.title('LOO for kNN: `cancer.csv`')

plt.subplot(212)
plt.plot(np.arange(0, maxK) + 1, s_loo_res, 'r-')
plt.axis([1, maxK, np.min(s_loo_res), np.max(s_loo_res)])
plt.xlabel('Number of neighbors')
plt.ylabel('LOO-metrics')
plt.title('LOO for kNN: `spam.csv`')

plt.show()
print("OK")
