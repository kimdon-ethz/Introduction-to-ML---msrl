#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.1"
__maintainer__ = "Jonas Lussi"
__email__ = "jlussi@ethz.ch"

"""Brief: Loads a csv File with input Vectors X and corresponding outputs y. Performs Ridge Regression with different
 regularization parameters on 10 Fold-split of the data. Reports average RMS over all splits for each regularization param
"""


import numpy as np
from sklearn.linear_model import Ridge,Lasso, SGDRegressor, ElasticNetCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV


#set display options
pd.options.display.max_columns = 999

#Load Train Data
df = pd.read_csv(os.path.join('..','data','train.csv'))


# restructure data

#Format Train Data
data_X=df.drop(["Id", "y"],axis = 1)
data_y=df['y']
#standardize the input vectors (axis=0, so it's along the columns), this is left out, as the test data is probably not standardized
#data_X=(data_X-data_X.mean(axis=0))/data_X.std(axis=0)

#Create feature transformation
X_features = pd.concat([data_X, data_X**2, np.exp(data_X), np.cos(data_X), pd.DataFrame(np.ones(data_X.shape[0]))], axis = 1)

#Create object which provides train/test indices to split data in train/test sets. Set random state for repeatability
# 700 for Leave-One-Out cross-validation.
kf = KFold(700, shuffle=True, random_state=42)

# set reg_param alpha as log space to get rough estimate in what range it should be
#reg_param=np.logspace(-3,2,20)

# set reg_param alpha as linspace around the value found from earlier trial to find optimal reg_param
reg_param=np.linspace(0.01,0.1,1000)
l1_ratio=np.linspace(0,0.1,15)

#solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga','sag']
solvers = ['sag']

#Loop over all the regression parameters
for l1 in l1_ratio:
    # create empty RMS list for this solver instance
    average_RMS = []
    for param in reg_param:
        #Create object for Ridge Regression
        #R_Reg = Ridge(alpha=param,normalize=False,tol=1e100,max_iter=8000,solver=i)
        R_Reg = SGDRegressor(alpha=param,l1_ratio=l1, max_iter=10000, tol=1e-5,penalty="elasticnet",random_state=42)

        # Initialize RMS
        RMS = 0

        #Split the Data and perform regression; iterate over generator object kf.split(..)
        for train_index, test_index in kf.split(X_features):
            X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
            y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
            #fit regressor to current fold
            R_Reg.fit(X_train, y_train)
            #predict on the remaining test set in current fold
            y_pred= R_Reg.predict(X_test)

            #get MS in current fold,
            RMS=RMS+mean_squared_error(y_test, y_pred)

        #Calculate RMS and append RMS to list to see which lamda performs best
        average_RMS.append((RMS/700)**0.5)

    print(average_RMS)
    #find the best performing reg parameter according to cross validation:
    min_ind=average_RMS.index(min(average_RMS))
    opt_reg=reg_param[min_ind]
    print('optimal cost=',min(average_RMS))
    print('optimal alpha',opt_reg)


    #use that reg_param to train on all the data, to increase performance:
    #R_Reg = Ridge(alpha=opt_reg,normalize=False,max_iter=8000,tol=1e100,solver="sag")
    R_Reg=SGDRegressor(alpha=opt_reg,l1_ratio=l1, max_iter=10000, tol=1e-5,penalty="elasticnet",random_state=42)
    R_Reg.fit(X_features, data_y)

    print('coefs=',R_Reg.coef_)
    print('l1=',l1)
#print the coefficients
df = pd.DataFrame(R_Reg.coef_)
df.to_csv('../results/out_Task1b.csv', header= False, index=False)





#NOTE: the above Code could be simplified with the RidgeCV function, however one loses a bit of control. The code would
# be as follows:
#reg_param = np.linspace(1e-2, 1e2, 1e5)
#clf = RidgeCV(alphas=reg_param).fit(X_features, data_y)
#weights = clf.coef_
#print (weights)
#print (clf.alpha_)


# # Use ElasticNetCV to find proper ratio between l1 and l2 coefficients
# l1_val = np.logspace(-4,1,11)
# alps_2 = np.logspace(-3,2,10)
# els = ElasticNetCV(l1_ratio=l1_val, alphas=alps_2, cv=5, max_iter=1e5).fit(X_features,data_y)
# print('alpha=', els.alpha_, 'l1_val=', els.l1_ratio_)
# print(els.coef_)
