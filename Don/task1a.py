import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# import the data file
df = pd.read_csv('train.csv')
# assign X and y from the train dataset
X = df.drop(['Id', 'y'], axis = 1)
y = df.y

# split data into 10-fold
kf = KFold(10, True, 1)

alphas = [0.01, 0.1, 1, 10, 100]
result = pd.DataFrame()

#for loop for train each kfold sets to get RMSE mean values for each alpha values
for i in alphas:
    rg = Ridge(i)
    RMSE_sum = 0
    for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            rg.fit(X_train, y_train)
            y_pred = rg.predict(X_test)
            RMSE = mean_squared_error(y_test, y_pred)**0.5
            RMSE_sum = RMSE_sum + RMSE
            print('alpha=', i, RMSE)
    result = result.append(pd.DataFrame([[i,RMSE_sum/10]], \
                                        columns=['alpha','RMSE']))

#write result file
test = pd.concat([result['RMSE']], axis = 1)
test.to_csv('result.csv', header=False, index=False)
