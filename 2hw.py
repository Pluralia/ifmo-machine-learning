import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DBSCAN import dbscan
# noinspection PyUnresolvedReferences
from matplotlib.cm import rainbow

c_df = pd.read_csv('blobs.csv')
data = c_df.values.astype(np.float32)

min_points_in_cluster_list = [7, 8, 9, 10]
eps_list = [0.2, 0.21, 0.22, 0.23]
fig, axes = plt.subplots(nrows=len(min_points_in_cluster_list), ncols=len(eps_list), figsize=(14, 8))
plt.ion()
fig.tight_layout()
plt.show()
for i, min_cluster_size in enumerate(min_points_in_cluster_list):
    for j, eps in enumerate(eps_list):
        labels = dbscan(data, eps, min_cluster_size)
        cluster_num = np.unique(labels).shape[0] - 1
        colors = rainbow(np.linspace(0, 1, cluster_num))
        for p, l in zip(data, labels):
            if l != -1:
                axes[i, j].plot(p[0], p[1], '.', color=colors[l])
            else:
                axes[i, j].plot(p[0], p[1], 'k.')
        axes[i, j].set_title('eps:' + str(eps) + '|min size:' + str(min_cluster_size) + '|clusters:' + str(cluster_num))
        plt.pause(0.001)
plt.ioff()
plt.show()
