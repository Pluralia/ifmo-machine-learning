import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def r_squared(real_y, predict_y):
    return 1 - np.sum((real_y - predict_y) ** 2) / np.sum((real_y - real_y.mean()) ** 2)


power = 6

########################################################################################################################


def make_x(x, power=1):
    X = np.ones([x.shape[0], power + 1])
    for i in range(1, power + 1):
        X[:, i] = x ** i
    return X


# noisysine
df = pd.read_csv('noisysine.csv')
df = df.values.astype(np.float32)
data = df[:, 0]
labels = df[:, 1]

X, x, Y, y = train_test_split(data, labels, train_size=0.8, test_size=0.2)
X = make_x(X, power)
x = make_x(x, power)

W = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(Y)
predict = x.dot(W)
rsquare = r_squared(y, predict)
print(rsquare)


max_rsquare = 0
max_alpha = 0
r_square = np.empty(201)
for i in range(201):
    alpha = 0.01 * (i - 100)
    W_reg = (np.linalg.inv(X.T.dot(X) + alpha * np.eye(power + 1)).dot(X.T)).dot(Y)
    r_square[i] = r_squared(y, x.dot(W_reg))
    if r_square[i] < 0:
        r_square[i] = r_square[i - 1]
    if max_rsquare < r_square[i]:
        max_rsquare = r_square[i]
        max_alpha = alpha
print(max_rsquare, max_alpha)

plt.plot(0.01 * (np.arange(201) - 100), r_square, 'r')
plt.title(f'noisysine (power={power}): ' + "r-square: %.3f;\n" % rsquare + "regular r-square: %.3f " % max_rsquare + f'(alpha={max_alpha})')
plt.show()

alpha = max_alpha
W_reg = (np.linalg.inv(X.T.dot(X) + alpha * np.eye(power + 1)).dot(X.T)).dot(Y)

plt.plot(data, labels, '.')
plt.plot(data, make_x(data, power).dot(W), 'g')
plt.plot(data, make_x(data, power).dot(W_reg), 'r')
plt.show()

########################################################################################################################


def make_x(x, power=1):
    X = np.ones([x.shape[0], x.shape[1] * power + 1])
    for i in range(1, power + 1):
        X[:, 1 + (i - 1) * x.shape[1]:1 + i * x.shape[1]] = x ** i
    return X


# hydrodynamics
df = pd.read_csv('hydrodynamics.csv')
df = df.values.astype(np.float32)
data = df[:, :6]
labels = df[:, 6]

X, x, Y, y = train_test_split(data, labels, train_size=0.8, test_size=0.2)
X = make_x(X, power)
x = make_x(x, power)

W = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(Y)
rsquare = r_squared(y, x.dot(W))
print(rsquare)


max_rsquare = 0
max_alpha = 0
r_square = np.empty(201)
I = np.eye(X.shape[1] * power + 1)
for i in range(201):
    alpha = 0.01 * (i - 100)
    W_reg = (np.linalg.inv(X.T.dot(X) + alpha * I).dot(X.T)).dot(Y)
    r_square[i] = r_squared(y, x.dot(W_reg))
    if r_square[i] < 0:
        r_square[i] = r_square[i - 1]
    if max_rsquare < r_square[i]:
        max_rsquare = r_square[i]
        max_alpha = alpha
print(max_rsquare, max_alpha)

plt.plot(0.01 * (np.arange(201) - 100), r_square, 'r')
plt.title(f'hydrodynamics (power={power}): ' + "r-square: %.3f;\n" % rsquare + "regular r-square: %.3f " % max_rsquare + f'(alpha={max_alpha})')
plt.show()
