import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
c_df = pd.read_csv('noisysine.csv')
data = c_df.values.astype(np.float32)

x = data[:, 0]
y = data[:, 1]

X = make_x(x, power)
W = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
predict = X.dot(W)
old = r_squared(y, predict)
print(old)


max = 0
r_square = np.empty(201)
for i in range(201):
    alpha = 0.01 * (i - 100)
    W_reg = (np.linalg.inv(X.T.dot(X) + alpha * np.eye(power + 1)).dot(X.T)).dot(y)
    predict_reg = X.dot(W_reg)
    r_square[i] = r_squared(y, predict_reg)
    if r_square[i] < 0:
        r_square[i] = r_square[i - 1]
    # print(r_square, alpha)
    if max < r_square[i]:
        max = r_square[i]
        alpha_max = alpha
print(max, alpha_max)

plt.plot(0.01 * (np.arange(201) - 100), r_square, 'r')
plt.title(f'noisysine (power={power}): ' + "r-square: %.3f;\n" % old + "regular r-square: %.3f " % max + f'(alpha={alpha_max})')
plt.show()

# alfa = 0.1
alpha = alpha_max
W_reg = (np.linalg.inv(X.T.dot(X) + alpha * np.eye(power + 1)).dot(X.T)).dot(y)
predict_reg = X.dot(W_reg)

plt.plot(x, y, '.')
plt.plot(x, predict, 'g')
plt.plot(x, predict_reg, 'r')
plt.show()

########################################################################################################################


def make_x(x, power=1):
    X = np.ones([x.shape[0], x.shape[1] * power + 1])
    for i in range(1, power + 1):
        X[:, 1 + (i - 1) * x.shape[1]:1 + i * x.shape[1]] = x ** i
    return X


# hydrodynamics
c_df = pd.read_csv('hydrodynamics.csv')
data = c_df.values.astype(np.float32)

x = data[:, :6]
y = data[:, 6]


X = make_x(x, power)
W = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
predict = X.dot(W)
old = r_squared(y, predict)
print(old)


max = 0
r_square = np.empty(201)
I = np.eye(x.shape[1] * power + 1)
for i in range(201):
    alpha = 0.01 * (i - 100)
    W_reg = (np.linalg.inv(X.T.dot(X) + alpha * I).dot(X.T)).dot(y)
    predict_reg = X.dot(W_reg)
    r_square[i] = r_squared(y, predict_reg)
    if r_square[i] < 0:
        r_square[i] = r_square[i - 1]
    # print(r_square, alpha)
    if max < r_square[i]:
        max = r_square[i]
        alpha_max = alpha
print(max, alpha_max)

plt.plot(0.01 * (np.arange(201) - 100), r_square, 'r')
plt.title(f'hydrodynamics (power={power}): ' + "r-square: %.3f;\n" % old + "regular r-square: %.3f " % max + f'(alpha={alpha_max})')
plt.show()
