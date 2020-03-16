import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# import training set data
train_df = pd.read_csv('../data/train.csv')
train_y = train_df['y'].to_numpy()
train_x = train_df.drop(['Id','y'],axis=1).to_numpy()

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

# build model
reg = LinearRegression().fit(phi, train_y)
# calculate get weights
weights = reg.coef_
print 'weights: ', weights

# export prediction to csv file
df = pd.DataFrame(weights, columns = ['weights'])
df.to_csv('../results/weights_list_0.csv', header= False, index=False)
