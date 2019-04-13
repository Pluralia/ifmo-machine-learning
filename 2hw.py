import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import perceptron


def disp_data(name, data, labels, W):
    plt.figure()
    for i, label in enumerate(labels):
        if label == -1:
            plt.plot(data[i, 0], data[i, 1], 'go')
        else:
            plt.plot(data[i, 0], data[i, 1], 'bo')
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y = -(W[1] * x + W[0]) / W[2]
    plt.plot(x, y, '-r')
    plt.title(name)
    plt.pause(0.001)


df = pd.read_csv("blobs2.csv").values
labels = df[:, -1] * 2 - 1
data = df[:, 0:-1]
tr_data, te_data, tr_labels, te_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

perc = perceptron.Perceptron()
perc.build(tr_data, tr_labels, 20)
tr_name = "Accuracy of train: %.3f" % np.mean(perc.predict(tr_data) == tr_labels)
te_name = "Accuracy of test: %.3f" % np.mean(perc.predict(te_data) == te_labels)

plt.ion()

disp_data(tr_name, tr_data, tr_labels, perc.get_weights())
disp_data(te_name, te_data, te_labels, perc.get_weights())

plt.ioff()
plt.show()
print("OK")