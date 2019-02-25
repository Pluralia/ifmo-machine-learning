import numpy as np
import numpy.linalg as la


# 1
def rNN(r, x, data, labels):
    size = data.shape[0]
    distance = np.sqrt((data - x) ** 2).sum(1).reshape(size, 1)

    dist_labels = np.hstack([labels, distance])
    dist_labels = dist_labels[dist_labels[:, 1].argsort()]

    neib_num = np.unique(labels)
    neib_num = dict(zip(neib_num, np.zeros(neib_num.shape[0])))

    i = 0
    while dist_labels[i, 1] < r:
        neib_num[dist_labels[i, 0]] += 1
        i += 1

    return np.argmax(np.array(list(neib_num.values()))), neib_num


# 2
def getLOO(n, minR, maxR, data, labels):
    size = data.shape[0]

    data_idx = []
    for j in range(size):
        tmp_dist = np.zeros((size, 2))
        tmp_dist[:, 0] = np.arange(size)
        tmp_dist[:, 1] = la.norm(data - data[j], axis=1)
        tmp_dist = tmp_dist[tmp_dist[:, 1].argsort()]
        tmp_dist = tmp_dist[1:]
        data_idx.append(tmp_dist[tmp_dist[:, 1] <= maxR])

    unique_labels = np.unique(labels).shape[0]
    res = np.zeros(n)
    for k in range(n):
        rr = minR + (maxR - minR) * k / n
        for j, idx in enumerate(data_idx):
            if not idx.any():
                res[k] += 1
                continue
            curr_num = np.zeros(unique_labels)
            i = 0
            while idx[i, 1] <= rr:
                curr_num[labels[idx[i, 0].astype(np.int32)]] += 1
                i += 1
                if i == idx.shape[0]:
                    break
            if np.argmax(curr_num) != labels[j]:
                res[k] += 1
    return res / size
