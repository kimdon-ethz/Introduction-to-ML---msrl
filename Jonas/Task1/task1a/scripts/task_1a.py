import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.model_selection import KFold



#Load Train Data
train = pd.read_csv(os.path.join('..','data','train.csv'))
# restructure data

#Format Train Data
data_X=train.drop(["Id", "y"],axis = 1)
data_y=train['y']


#data_set_y = data_set_df['y'].to_numpy()
#data_set_x = data_set_df.drop(['Id', 'y'], axis=1).to_numpy()


# regularization parameter list
reg_param = [0.01, 0.1, 1, 10, 100]


#Create object which provides train/test indices to split data in train/test sets.
kf = KFold(10, True, 1)

#Split the Data
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#
# # split training data set into 10-fold and store them in a list
# k_folds = 10
# samples_per_fold = np.shape(data_set_x)[0] // k_folds
# data_folds_x = []
# data_folds_y = []
# for k in xrange(0, k_folds):
#     x_fold_k = data_set_x[k * samples_per_fold:(k + 1) * samples_per_fold]
#     y_fold_k = data_set_y[k * samples_per_fold:(k + 1) * samples_per_fold]
#     data_folds_x.append(x_fold_k)
#     data_folds_y.append(y_fold_k)
# # add data that remains to last set (data might not evenly divided into k folds,
# # example: 506 samples, k = 10 -> 9 data folds with 50 samples, last fold with 56 samples)
# data_remains_x = data_set_x[(k + 1) * samples_per_fold:]
# data_remains_y = data_set_y[(k + 1) * samples_per_fold:]
# data_folds_x[-1] = np.vstack((data_folds_x[-1], data_remains_x))
# data_folds_y[-1] = np.hstack((data_folds_y[-1], data_remains_y))
#
#
# # perform cross validation for ridge regression for every regularization parameter
# # looping over all regularization parameters
# RMSE_list = []
# for i in xrange(0, len(reg_param)):
#     # looping over all folds
#     RMSE_folds_list = []
#     for k in xrange(0, k_folds):
#         # seperate x and y data of each fold into test and validation sets
#         valid_x = data_folds_x[k]
#         valid_y = data_folds_y[k]
#         test_x = np.vstack((data_folds_x[:k] + data_folds_x[k+1 :]))
#         test_y = np.hstack((data_folds_y[:k] + data_folds_y[k+1 :]))
#         # fit regression model to all test folds
#         clf = Ridge(alpha=reg_param[i])
#         clf.fit(test_x, test_y)
#         # make prediction on validation fold
#         predict_y = clf.predict(valid_x)
#         # evaluate performance performance of prediction obtained test fold with ground truth of validation fold
#         RMSE_fold = mean_squared_error(valid_y, predict_y) ** 0.5
#         # store RMSE of individual folds in list
#         RMSE_folds_list.append(RMSE_fold)
#     # average folds from and add to new list
#     RMSE_list.append(np.mean(RMSE_folds_list))
#
# # export prediction to csv file
# df = pd.DataFrame(RMSE_list, columns = ['RMSE'])
# df.to_csv('../result/RMSE_list_1.csv', header= False, index=False)
