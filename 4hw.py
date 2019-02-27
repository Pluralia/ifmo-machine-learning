import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KMeans import purity

c_df = pd.read_csv('cancer.csv')
c_df = c_df.values

c_labels = c_df[:, 0]
c_data = c_df[:, 1:].astype(np.float32)

c_labels_dict = dict(zip(np.unique(c_labels), np.arange(c_labels.shape[0])))
print(c_labels_dict)
c_labels = np.array(list(map(lambda x: c_labels_dict[x], c_labels)))

purity_res_nonorm = purity(2, 10, c_data, c_labels, do_radius_norm=False)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
ax0.plot(np.arange(2, 11), purity_res_nonorm, 'b-')
ax0.set_xlabel('Number of cluster')
ax0.set_ylabel('purity-metrics')
ax0.set_title('purity for NMeans no norm: `cancer.csv`')

purity_res_norm = purity(2, 10, c_data, c_labels, do_radius_norm=True)

ax1.plot(np.arange(2, 11), purity_res_norm, 'b-')
ax1.set_xlabel('Number of cluster')
ax1.set_ylabel('purity-metrics')
ax1.set_title('purity for NMeans with norm: `cancer.csv`')
plt.show()
