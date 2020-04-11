#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.1"
__maintainer__ = "Jonas Lussi"
__email__ = "jlussi@ethz.ch"

"""Brief: Loads a csv File with input Vectors X and corresponding outputs y. Performs Ridge Regression with different
 regularization parameters on 10 Fold-split of the data. Reports average RMS over all splits for each regularization param
"""


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

#Load Train Data
df = pd.read_csv(os.path.join('..','data','train.csv'))
# restructure data

#Format Train Data
data_X=df.drop(["Id", "y"],axis = 1)
#standardize the input vectors (axis=0, so it's along the columns)
#data_X=(data_X-data_X.mean(axis=0))/data_X.std(axis=0)

data_y=df['y']

# regularization parameter list
reg_param = [0.01, 0.1, 1, 10, 100]

#Create object which provides train/test indices to split data in train/test sets. Set random state for repeatability
kf = KFold(10, shuffle=True, random_state=42)

#create RMS list
average_RMS=[]

#Loop over all the regression parameters
for param in reg_param:
    #Create object for Ridge Regression
    R_Reg = Ridge(alpha=param,normalize=False)

    # Initialize RMS
    RMS = 0

    #Split the Data and perform regression; iterate over generator object kf.split(..)
    for train_index, test_index in kf.split(data_X):
        X_train, X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        #fit regressor to current fold
        R_Reg.fit(X_train, y_train)
        #predict on the remaining test set in current fold
        y_pred= R_Reg.predict(X_test)
        #get RMS in current fold
        RMS=RMS+mean_squared_error(y_test, y_pred) ** 0.5

    #Append average RMS to list for writing to csv
    average_RMS.append(RMS/10)

#Write to file
myFrame = pd.DataFrame(average_RMS)
myFrame.to_csv('../results/outTask1.csv', index=False,header=False)
