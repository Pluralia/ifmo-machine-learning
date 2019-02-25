import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from rNN import getLOO


# 4
n = 20
minR = 0.01
maxR = 2
scaler = MinMaxScaler()

c_df = pd.read_csv('cancer.csv')
c_df = c_df.values

c_labels = c_df[:, 0]
c_data = c_df[:, 1:].astype(np.float)
scaler.fit(c_data)
c_data = scaler.transform(c_data)

c_labels_dict = dict(zip(np.unique(c_labels), np.arange(c_labels.shape[0], dtype=np.int32)))
# print(c_labels_dict)
c_labels = np.array([c_labels_dict[x] for x in c_labels])

c_loo_res = getLOO(n, minR, maxR, c_data, c_labels)
# print(c_loo_res)

radius = minR + (maxR - minR) * np.arange(n) / n
plt.plot(radius, c_loo_res, 'b-')
plt.xlabel('Radius')
plt.ylabel('LOO-metrics')
plt.title('LOO for kNN: `cancer.csv`')


# n = 10
# minR = 0.01
# maxR = 1
# scaler = MinMaxScaler()
#
# s_df = pd.read_csv('spam.csv')
# s_df = s_df.values
#
# s_labels = s_df[:, -1]
# s_data = s_df[:, 0:-1]
# scaler.fit(s_data)
# s_data = scaler.transform(s_data)
#
# s_labels_dict = dict(zip(np.unique(s_labels), np.arange(s_labels.shape[0])))
# # print(s_labels_dict)
# s_labels = np.array(list(map(lambda x: s_labels_dict[x], s_labels)))
#
# s_loo_res = getLOO(n, minR, maxR, s_data, s_labels)
# # print(s_loo_res)
#
# radius = minR + (maxR - minR) * np.arange(n) / n
# plt.plot(radius, s_loo_res, 'r-')
# plt.xlabel('Radius')
# plt.ylabel('LOO-metrics')
# plt.title('LOO for kNN: `spam.csv`')

plt.show()
print("OK")
