import numpy as np
import matplotlib.pyplot as plt
from kNN import kNN


# 1
x1_size = 100
x2_size = 200

X1 = np.random.randn(x1_size, 2)
X2 = 0.9 * np.random.randn(x2_size, 2) + 3
data = np.vstack([X1, X2])
labels = np.vstack([np.zeros((x1_size, 1)), np.ones((x2_size, 1))])

plt.plot(data[:x1_size, 0], data[:x1_size, 1], 'y.')
plt.plot(data[x1_size:, 0], data[x1_size:, 1], 'c.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data set visualisation: normal distribution')


num_of_elements = 100
x = np.random.randn(num_of_elements, 2) + 1

for i in range(num_of_elements):
    x_label, _ = kNN(7, x[i], data, labels)
    print(x_label, "|", x[i])
    if x_label == 0:
        plt.plot(x[i, 0], x[i, 1], 'go')
    else:
        plt.plot(x[i, 0], x[i, 1], 'bo')

plt.show()
print("OK")
