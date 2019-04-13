import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import perceptron


def extend_data_core(data, dims, max_dims):
    new_data = data
    x = data[:, 0]
    y = data[:, 1]
    for i in range(dims + 1):
        add_data = (x ** (dims - i) * y ** i)
        add_data = np.expand_dims(add_data, axis=1)
        new_data = np.hstack((new_data, add_data))
    if dims != max_dims:
        new_data = extend_data_core(new_data, dims + 1, max_dims)
    return new_data


def extend_data(data, dims):
    return extend_data_core(data, 2, dims)

def disp_data(name, data, labels, W, dims):
    plt.figure()
    for i, label in enumerate(labels):
        if label == -1:
            plt.plot(data[i, 0], data[i, 1], 'go')
        else:
            plt.plot(data[i, 0], data[i, 1], 'bo')
    # y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    # x = (-W[1] - W[4] * y + np.sqrt((W[1] + W[4] * y) ** 2 - 4 * W[3] * (W[0] + W[2] * y + W[5] * y ** 2))) / (2 * W[3])
    # plt.plot(x, y, '-r')
    # x = (-W[1] - W[4] * y - np.sqrt((W[1] + W[4] * y) ** 2 - 4 * W[3] * (W[0] + W[2] * y + W[5] * y ** 2))) / (2 * W[3])
    size = 1100
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), size)
    y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), size)
    xx, yy = np.meshgrid(x, y)
    xy = np.hstack([xx.reshape([size*size, 1]), yy.reshape([size*size, 1])])
    exp_data = extend_data(xy, dims)
    one = np.ones([size*size, 1])
    plot_data = np.hstack([one, exp_data])
    wexp = np.expand_dims(W, axis=1)
    plot_vector = plot_data.dot(wexp)
    z = plot_vector.reshape([size,size])

    eps = 1
    xx, yy = [], []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if abs(z[i, j]) < eps:
                xx.append(x[j])
                yy.append(y[i])
    plt.scatter(xx, yy, c='r', s=2)
    # plt.plot(x, y, '-r')
    plt.title(name)
    plt.pause(0.001)


df = pd.read_csv("blobs2.csv").values
labels = df[:, -1] * 2 - 1
data = df[:, 0:-1]

dims = 4
data = extend_data(data, dims)
tr_data, te_data, tr_labels, te_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

perc = perceptron.Perceptron()
perc.build(tr_data, tr_labels, 20)
tr_name = f"Accuracy of train ({dims}): %.3f" % np.mean(perc.predict(tr_data) == tr_labels)
te_name = f"Accuracy of test ({dims}): %.3f" % np.mean(perc.predict(te_data) == te_labels)

plt.ion()

disp_data(tr_name, tr_data, tr_labels, perc.get_weights(), dims)
disp_data(te_name, te_data, te_labels, perc.get_weights(), dims)

plt.ioff()
plt.show()
print("OK")