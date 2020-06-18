import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn import linear_model
import os
from IPython.display import display
import csv

import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import mean_squared_error

#Load Train Data
train = pd.read_csv(os.path.join('..','task0_sl19d1','train.csv'))
#Format Train Data
X=train.drop(["Id", "y"],axis = 1)
y=train['y']
#Create Regressor and train on Train-Data
reg = LinearRegression().fit(X, y)
#Test performance on training set to verify calculation (should be good)
y_pred=reg.predict(X)
RMSE = mean_squared_error(y, y_pred)**0.5
print(RMSE)

#and now predict on test set:
#load test Data
test = pd.read_csv(os.path.join('..','task0_sl19d1','test.csv'))
#format test Data
X_test=test.drop(["Id"],axis = 1)
#predict test Data and format
y_pred=pd.DataFrame({'y':reg.predict(X_test)})
#concate Id and y into a single vector to write to csv file
myFrame=pd.concat([test['Id'],y_pred],axis=1)
#write to csv
myFrame.to_csv('out.csv', index=False)






#reg = linear_model.LinearRegression()  # create object for the class
#reg.fit(X, Y)  # perform linear regression
#Y_pred = reg.predict(X)  # make predictions on labeled data

