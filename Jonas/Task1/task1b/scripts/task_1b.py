#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.1"
__maintainer__ = "Jonas Lussi"
__email__ = "jlussi@ethz.ch"

"""Brief: Loads a csv File with input Vectors X and corresponding outputs y. Performs Feature transformation,
 Ridge Regression with parameter optimization with Crossvalidation of the data. Prints the optimal weights for 
 regression to a csv file.
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

#Format Train Data
data_X=df.drop(["Id", "y"],axis = 1)
data_y=df['y']

#standardize the input vectors (axis=0, so it's along the columns), this is left out, as the test data is probably not standardized
#data_X=(data_X-data_X.mean(axis=0))/data_X.std(axis=0)

#Create feature transformation
X_features = pd.concat([data_X, data_X**2, np.exp(data_X), np.cos(data_X), pd.DataFrame(np.ones(data_X.shape[0]))], axis = 1)

reg_param = np.linspace(1e-2, 1e2, 1e4)
clf = RidgeCV(alphas=reg_param).fit(X_features, data_y)
weights = clf.coef_
print (weights)
print (clf.alpha_)



#Create object which provides train/test indices to split data in train/test sets. Set random state for repeatability
# 700 would be Leave-One-Out cross-validation. 100 is good tradeoff between speed and variance/bias
nfolds=100
kf = KFold(nfolds, shuffle=False, random_state=1)

# set reg_param alpha as log space to get rough estimate in what range it should be
#  (now commented out, only for initial guess)
#reg_param=np.logspace(-3,2,20)

# set reg_param alpha as linspace around the value found from earlier trial with logspace to find optimal reg_param
reg_param=np.linspace(19,23,10)

# Regression: Note that we use pure Ridge regression, as Lasso or combined Lasso and Ridge lead to worse results

# List the different solvers that have to be tested for optimal performance
solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga','sag']

# create empty RMS dictionaries for the solver instances
average_RMS = {}
opt_reg= {}

#Loop over all the regression parameters
for i in solvers:
    #create empty list for that key in the dictionary
    average_RMS[i] = []

    for param in reg_param:
        #Create object for Ridge Regression
        R_Reg = Ridge(alpha=param,normalize=False,tol=1e-3,max_iter=6000,solver=i, random_state=1234)

        # Initialize RMS
        MSE = 0

        #Split the Data and perform regression; iterate over generator object kf.split(..)
        for train_index, test_index in kf.split(X_features):
            X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
            y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
            #fit regressor to current fold
            R_Reg.fit(X_train, y_train)
            #predict on the remaining test set in current fold
            y_pred= R_Reg.predict(X_test)

            #get MSE in current fold, and add them up for all folds to get total RMSE, like this it's closer to RMSE
            #calculation over whole dataset
            MSE=MSE+mean_squared_error(y_test, y_pred)

        #Calculate RMS and append RMS to list to see which lamda performs best
        average_RMS[i].append((MSE/nfolds)**0.5)

    #find the best performing reg parameter according to cross validation:
    min_ind=average_RMS[i].index(min(average_RMS[i]))
    opt_reg[i]=reg_param[min_ind]
    print(i,":")
    print('optimal cost=',min(average_RMS[i]))
    print('optimal alpha',opt_reg[i])


#use that reg_param to train on all the data, to increase performance:
R_Reg = Ridge(alpha=opt_reg['sag'],normalize=False,max_iter=6000,tol=1e-3,solver="sag",random_state=1234)
R_Reg.fit(X_features, data_y)

print('coefs=',R_Reg.coef_)
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


# # This was used to see the impact of combining L1 (Lasso) and L2 (Ridge) regularization
# Use ElasticNetCV to find proper ratio between l1 and l2 coefficients
# l1_val = np.logspace(-4,1,11)
# alps_2 = np.logspace(-3,2,10)
# els = ElasticNetCV(l1_ratio=l1_val, alphas=alps_2, cv=5, max_iter=1e5).fit(X_features,data_y)
# print('alpha=', els.alpha_, 'l1_val=', els.l1_ratio_)
# print(els.coef_)
# --> L1 almost 0
