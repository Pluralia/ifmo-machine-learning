import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def apply(tr_data, te_data, tr_labels, te_labels, kernel, degree=1):
    plt.figure()
    clf = SVC(gamma='scale', kernel=kernel, degree=degree, probability=True)
    clf.fit(tr_data, tr_labels)

    res = clf.predict(te_data)
    err = round(1 - sum(res == te_labels) / te_labels.shape[0], 3)
    title = f'{kernel}:{degree}, err={err}'
    print(title)

    res = np.array(['g.' if l == 1 else 'b.' for l in res])
    for idx, color in enumerate(res):
        plt.plot(te_data[idx, 0], te_data[idx, 1], color)
        for sup_idx in clf.support_:
            plt.plot(tr_data[sup_idx, 0], tr_data[sup_idx, 1], 'yx' if tr_labels[sup_idx] == 1 else 'cx')

    n = 1000
    x = np.linspace(np.min(te_data[:, 0]), np.max(te_data[:, 0]), n)
    y = np.linspace(np.min(te_data[:, 1]), np.max(te_data[:, 1]), n)
    grid = np.array([x for x in it.product(x, y)])
    z = grid[np.abs(clf.predict_proba(grid)[:, 0] - 0.5) < 0.01]
    plt.plot(z[:, 0], z[:, 1], 'r.')

    plt.title(title)
    plt.pause(0.0001)


df = pd.read_csv("blobs2.csv").values
labels = df[:, -1] * 2 - 1
data = df[:, 0:-1]

methods = [('linear', [1]), ('poly', [2, 3, 5]), ('rbf', [1])]
tr_data, te_data, tr_labels, te_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

plt.ion()

for (kernel, degree) in methods:
    for deg in degree:
        apply(tr_data, te_data, tr_labels, te_labels, kernel, deg)


plt.ioff()
plt.show()
