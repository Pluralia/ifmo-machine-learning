import collections
import numpy as np
import math


# train labels
def prior_prob(labels):
    labels_dict = collections.Counter(labels)
    res = np.zeros(2)
    for i in range(0, 2):
        res[i] = labels_dict[i] / labels.shape[0]
    return res


# train data & labels
def mean_variance(data, labels):
    features_num = data.shape[1]
    mean = np.zeros((2, features_num))
    var = np.zeros((2, features_num))
    n = np.bincount(labels)

    x0 = np.zeros((n[0], features_num))
    x1 = np.zeros((n[1], features_num))
    size = data.shape[0]
    k = 0
    for i in range(0, size):
        if labels[i] == 0:
            x0[k] = data[i]
            k += 1
    k = 0
    for i in range(0, size):
        if labels[i] == 1:
            x1[k] = data[i]
            k += 1

    for j in range(0, features_num):
        mean[0][j] = np.mean(x0.T[j])
        var[0][j] = np.var(x0.T[j])
        mean[1][j] = np.mean(x1.T[j])
        var[1][j] = np.var(x1.T[j])
    return mean, var


# validate data
def posterior_prob(mean, var, data):
    size = data.shape[0]
    features_num = data.shape[1]
    res = np.ones((size, 2))
    for k in range(0, size):
        for i in range(0, 2):
            for j in range(0, features_num):
                if var[i][j] == 0:
                    print("ZERO")
                    res[k, i] = 1
                else:
                    res[k, i] *= 1 / math.sqrt(2 * math.pi * var[i][j])
                    res[k, i] *= math.exp(-0.5 * pow(data[k, j] - mean[i][j], 2) / var[i][j])
    return res


# train data & labels + validate data
def predict(tr_data, tr_labels, val_data):
    size = val_data.shape[0]
    prior = prior_prob(tr_labels)
    mean, var = mean_variance(tr_data, tr_labels)
    posterior = posterior_prob(mean, var, val_data)
    labels = np.zeros(size)
    feature_prob = np.zeros(size)
    for k in range(0, size):
        total_prob = 0
        for j in range(0, 2):
            total_prob += posterior[k, j] * prior[j]
        prob = np.ones(2)
        for i in range(0, 2):
            if total_prob == 0:
                prob[i] = 0
            else:
                prob[i] = (posterior[k, i] * prior[i]) / total_prob
        labels[k] = int(prob.argmax())
        feature_prob[k] = prob[1]
    return labels, feature_prob
