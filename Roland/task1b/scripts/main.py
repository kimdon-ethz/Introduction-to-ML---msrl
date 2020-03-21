import numpy as np
from sklearn.linear_model import RidgeCV
import pandas as pd

# import training set data
train_df = pd.read_csv('../data/train.csv')
train_y = train_df['y'].to_numpy()
train_x = train_df.drop(['Id', 'y'], axis=1).to_numpy()

# update non-linear features of basis function
phi_lin = train_x
phi_quad = train_x**2
phi_exp = np.exp(train_x)
phi_cos = np.cos(train_x)
phi_const = np.ones(np.shape(train_x)[0])

phi = np.hstack((phi_lin,
                 phi_quad,
                 phi_exp,
                 phi_cos,
                 np.reshape(phi_const, (np.shape(phi_const)[0], 1)))
                )

reg_param = np.linspace(1e-2, 1e2, 1e4)
clf = RidgeCV(alphas=reg_param).fit(phi, train_y)
weights = clf.coef_
print weights
print clf.alpha_


# export prediction to csv file
df = pd.DataFrame(weights, columns=['weights'])
df.to_csv('../results/result.csv', header= False, index=False)