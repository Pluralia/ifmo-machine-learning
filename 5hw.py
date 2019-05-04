import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso


# noisysine
df = pd.read_csv('noisysine.csv')
df = df.values.astype(np.float32)

data = df[:, 0]
labels = df[:, 1]
data = data.reshape(data.shape[0], 1)

# hydrodynamics
# df = pd.read_csv('hydrodynamics.csv')
# df = df.values.astype(np.float32)
#
# data = df[:, :6]
# labels = df[:, 6]

tr_data, val_data, tr_labels, val_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

best_alpha = 0
best_deg = 0
rscore = 0
for deg in range(1, 6):
    print(f'degree={deg} ----------------------------------------------------------------------------------------------')
    for i in range(1, 11):
        alpha = i * 0.1
        print(f'alpha={alpha}')
        steps = [
            ('scalar', StandardScaler()),
            ('poly', PolynomialFeatures(degree=deg)),
            ('model', Lasso(alpha=alpha, max_iter=10000000))
        ]

        lasso_pipe = Pipeline(steps)
        lasso_pipe.fit(tr_data, tr_labels)

        print('Training score: {}'.format(lasso_pipe.score(tr_data, tr_labels)))
        print('Validate score: {}'.format(lasso_pipe.score(val_data, val_labels)))
        print("")

        score = lasso_pipe.score(val_data, val_labels)
        if rscore < score:
            rscore = score
            best_alpha = alpha
            best_deg = deg

print(f'best_alpha={best_alpha}, best_degree={best_deg}, r-score={rscore}')
