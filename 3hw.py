import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Agglomerative import AgglomerativeClustering
# noinspection PyUnresolvedReferences
from matplotlib.cm import rainbow

c_df = pd.read_csv('blobs.csv')
data = c_df.values.astype(np.float32)

clustering = AgglomerativeClustering(data)

experiment_number = 16
rows = int(np.sqrt(experiment_number))
fig, axes = plt.subplots(nrows=rows, ncols=experiment_number//rows, figsize=(14, 8))
plt.ion()
fig.tight_layout()
plt.show()
for cluster_number in range(experiment_number):
    labels = clustering.get_labels(cluster_number + 1)
    colors = rainbow(np.linspace(0, 1, cluster_number + 1))
    i = cluster_number // rows
    j = cluster_number - i * rows
    for p, l in zip(data, labels):
        axes[i, j].plot(p[0], p[1], '.', color=colors[l])
    axes[i, j].set_title('clusters:' + str(cluster_number + 1))
    plt.pause(0.001)
plt.ioff()
plt.show()
