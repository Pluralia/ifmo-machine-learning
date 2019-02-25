import numpy as np
import numpy.linalg as la


# 1
def kNN(k, x, data, labels):
    size = data.shape[0]
    distance = np.sqrt((data - x) ** 2).sum(1).reshape(size, 1)

    dist_labels = np.hstack([labels, distance])
    dist_labels = dist_labels[dist_labels[:, 1].argsort()][:k, 0]

    neib_num = np.unique(labels)
    neib_num = dict(zip(neib_num, np.zeros(neib_num.shape[0])))

    for i in range(k):
        neib_num[dist_labels[i]] += 1

    return np.argmax(np.array(list(neib_num.values()))), neib_num


# 2
def getLOO(k, data, labels):
    size = data.shape[0]

    data_idx = np.zeros((size, k+1))
    for j in range(size):
        tmp_dist = np.zeros((size, 2))
        for i in range(size):
            tmp_dist[i, 0] = i
            tmp_dist[i, 1] = la.norm(data[j] - data[i])
        tmp_dist = tmp_dist[tmp_dist[:, 1].argsort()][0:k+1, 0]
        for i in range(k+1):
            data_idx[j, i] = labels[int(tmp_dist[i])]

    res = np.zeros(k)
    for kk in range(1, k+1):
        for j in range(size):
            curr_num = np.zeros(np.unique(labels).shape[0])
            for i in range(1, kk+1):
                curr_num[data_idx[j, i].astype(np.int32)] += 1
            if np.argmax(curr_num) != data_idx[j, 0]:
                res[kk-1] += 1
    return list(map((lambda x: x / size), res))
