import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score

# import the data file
df = pd.read_csv('train.csv')

# assign X and y from the train dataset
X = df.drop(['Id', 'y'], axis=1)
y = df.y

# Feature transformation of x
const = pd.DataFrame(np.ones(X.shape[0]))
X_tf = pd.concat([X, X**2, np.exp(X), np.cos(X), const], axis = 1)
X_tf.columns = ['x'+str(i) for i in range(1, 22)]

# Use ElasticNetCV to find proper ratio between l1 and l2 coefficients
l1_val = np.logspace(-7,-2,10)
alps_2 = np.linspace(0.028,0.033,500)
els = ElasticNetCV(l1_ratio=l1_val, alphas=alps_2, cv=10, max_iter=1e5).fit(X_tf,y)
print('alpha=', els.alpha_, 'l1_val=', els.l1_ratio_)
print(els.coef_)
# l1_ratio almost converges to zero, which means that the regression is nearly l2 penalty
   

#write result file
test = pd.concat([pd.DataFrame(els.coef_)], axis = 1)
test.to_csv('result.csv', header=False, index=False)
