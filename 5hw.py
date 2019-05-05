import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


power = 5
print(f'power={power}')
print('###############################################################################################################')
########################################################################################################################


def make_x(x, power=1):
    X = np.ones([x.shape[0], power + 1])
    for i in range(1, power + 1):
        X[:, i] = x ** i
    return X


# noisysine
print('NOISYSINE')
df = pd.read_csv('noisysine.csv')
df = df.values.astype(np.float32)
data = df[:, 0]
labels = df[:, 1]
coef = np.zeros(power + 1)

X, x, Y, y = train_test_split(data, labels, train_size=0.8, test_size=0.2)
X = make_x(X, power)
x = make_x(x, power)

alpha_num = 10
rscore = np.zeros(10)
best_rscore = 0
best_alpha = 0
for a in range(1, alpha_num + 1):
    alpha = a * 0.1
    print(f'alpha={alpha}')
    clf = Lasso(alpha=alpha, max_iter=1000000)
    clf.fit(X, Y)
    print(f'Coefficients: {clf.coef_}')
    rscore[a - 1] = clf.score(X, Y)
    if rscore[a - 1] > best_rscore:
        best_rscore = rscore[a - 1]
        best_alpha = alpha
        coef = clf.coef_

plt.plot(np.arange(1, alpha_num + 1) * 0.1, rscore, 'r')
plt.title(f'noisysine (power={power}): ' + "r-score=%.3f, " % clf.score(x, y) + f'alpha={best_alpha}\n')
plt.show()

########################################################################################################################


def make_x(x, power=1):
    X = np.ones([x.shape[0], x.shape[1] * power + 1])
    for i in range(1, power + 1):
        X[:, 1 + (i - 1) * x.shape[1]:1 + i * x.shape[1]] = x ** i
    return X


# hydrodynamics
print('HYDRODYNAMICS')
df = pd.read_csv('hydrodynamics.csv')
df = df.values.astype(np.float32)
data = make_x(df[:, :6], power)
labels = df[:, 6]
coef = np.zeros(X.shape[1])

X, x, Y, y = train_test_split(data, labels, train_size=0.8, test_size=0.2)
X = make_x(X, power)
x = make_x(x, power)

alpha_num = 10
rscore = np.zeros(10)
best_rscore = 0
best_alpha = 0
for a in range(1, alpha_num + 1):
    alpha = a * 0.1
    print(f'alpha={alpha}')
    clf = Lasso(alpha=alpha, max_iter=10000)
    clf.fit(X, Y)
    print(f'Coefficients: {clf.coef_}')
    rscore[a - 1] = clf.score(X, Y)
    if rscore[a - 1] > best_rscore:
        best_rscore = rscore[a - 1]
        best_alpha = alpha
        coef = clf.coef_

plt.plot(np.arange(1, alpha_num + 1) * 0.1, rscore, 'r')
plt.title(f'hydrodynamics (power={power}): ' + "r-score=%.3f, " % clf.score(x, y) + f'alpha={best_alpha}\n')
plt.show()
