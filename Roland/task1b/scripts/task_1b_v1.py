import numpy as np
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

# import training set data
train_df = pd.read_csv('../data/train.csv')
train_y = train_df['y'].to_numpy()
train_x = train_df.drop(['Id', 'y'], axis=1).to_numpy()

# standardize training set to ensure that each feature has zero mean and unit variance
# scaler = StandardScaler()
# scaler.fit(train_x)
# train_x = scaler.transform(train_x)

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

# regularization parameter list
reg_param = np.linspace(1e-3, 1e3, 100)
kf = KFold(n_splits=10)
kf.get_n_splits(phi)
w_best_fold_list = []
RMSE_reg_param_list = []
w_fold_list = []

# optimize for the best regression parameter
for i in xrange(0, len(reg_param)):
    w_list = []
    RMSE_folds_list = []

    # make prediction on all folds, check performance, pick the weights of the most performing fold
    for train_index, test_index in kf.split(phi):
        # split set into folds
        phi_train, phi_valid = phi[train_index], phi[test_index]
        y_train, y_valid = train_y[train_index], train_y[test_index]
        # fit ridge regression model to training folds
        clf = Ridge(alpha=reg_param[i])
        clf.fit(phi_train, y_train)
        # make prediction on validation fold
        predict_y = clf.predict(phi_valid)
        # evaluate performance performance of prediction obtained test fold with ground truth of validation fold
        RMSE_fold = mean_squared_error(y_valid, predict_y) ** 0.5
        # store RMSE of individual folds in list
        RMSE_folds_list.append(RMSE_fold)
        # store weights
        w_list.append(clf.coef_)

    # store RMSE of individual reg parameters in list
    RMSE_reg_param_list.append(RMSE_folds_list)
    # store weights
    w_fold_list.append(w_list)

# pick the regression parameter with the smallest RMSE
w_best_fold_list.append(w_list[np.argmin(RMSE_folds_list)])
w_best_fold_list = w_best_fold_list[0]
print min(RMSE_folds_list)

# plot RMSE
plt.plot(RMSE_reg_param_list)
plt.plot(np.amax(RMSE_folds_list))
plt.xlabel("regression parameter")
plt.ylabel("RMSE")
# export prediction to csv file
df = pd.DataFrame(w_best_fold_list, columns=['weights'])
df.to_csv('../results/weights_list_7.csv', header= False, index=False)

plt.show()