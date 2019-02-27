import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KMeans import kmeans

c_df = pd.read_csv('blobs.csv')
data = c_df.values.astype(np.float32)

cluster_number = 6
labels, models = kmeans(data, cluster_number)

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'silver', 'lime']
for i, p in enumerate(data):
    plt.plot(p[0], p[1], '.', color=colors[labels[i]])
for m in models:
    plt.plot(m[0], m[1], "^k")

plt.title('`blobs.csv` data set visualisation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
