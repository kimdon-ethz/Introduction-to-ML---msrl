import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# import the data file
df = pd.read_csv('train.csv')
# assign X and y from the train dataset
X = df.drop(['Id', 'y'], axis = 1)
y = df.y

scaler = StandardScaler()
scaler.fit(X)
X_s = pd.DataFrame(scaler.transform(X))
X_s.columns = ['x1', 'x2', 'x3', 'x4', 'x5']

# Feature transformation of x
ones = pd.DataFrame(np.ones(X_s.shape[0]))
X_tf = pd.concat([X_s, X_s**2, np.exp(X_s), np.cos(X_s), ones], axis = 1)
print(X_tf)

regr = LinearRegression()
regr.fit(X_tf, y)

print(regr.coef_)

#write result file
#test = pd.concat([pd.DataFrame(regr.coef_)], axis = 1)
#test.to_csv('result.csv', header=False, index=False)
