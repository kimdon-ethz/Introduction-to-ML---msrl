#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

data_train = pd.read_csv('../data/train.csv')
data_predict = pd.read_csv('../data/test.csv')
data_train

print('Value count:')
print(data_train.Active.value_counts())
print('%.2f %% of all proteins are active' % \
     (data_train.Active.value_counts()[1]/
     data_train.Active.value_counts()[0]*100))

def seperate_sites(data):
    
    # feature engineering: every site seperate feature
    sites = pd.DataFrame({'site_0':[],
                          'site_1':[],
                          'site_2':[],
                          'site_3':[]})
    i = 0
    for sequence in  data['Sequence']:
        sites.loc[i,'site_0'] = sequence[0]
        sites.loc[i,'site_1'] = sequence[1]
        sites.loc[i,'site_2'] = sequence[2]
        sites.loc[i,'site_3'] = sequence[3]
        i += 1
        
    return sites

def feature_extraction(data):
    # turn X into dict
    # turn each row as key-value pairs
    X_dict = data.to_dict(orient='records')
    # instantiate a Dictvectorizer object for X
    dv_X = DictVectorizer(sparse=False) 
    # sparse = False makes the output is not a sparse matrix
    # apply dv_X on X_dict
    X_encoded = dv_X.fit_transform(X_dict)

    return X_encoded

# Separate majority and minority classes
majority = data_train[data_train.Active==0]
minority = data_train[data_train.Active==1]

# Upsample minority class to match majority class
minority_upsampled = resample(minority, 
                                 replace=True,
                                 n_samples=majority.Active.value_counts()[0],
                                 random_state=123)
# Combine majority class with upsampled minority class
data_train = pd.concat([majority, minority_upsampled])
# shuffle rows
data_train = data_train.sample(frac=1)
data_train.Active.value_counts()

# # reduce data size for faster prototyping
# data_train = data_train.iloc[0:5000,:]
# data_train.head()

# seperate features from labels
y = data_train.drop(['Sequence'], axis=1).values.ravel()
    
sites_train = seperate_sites(data_train)

# # export seperated features
# sites_train.to_csv('../data/sites_train.csv')

# # import seperated data again
# sites_train = pd.read_csv('../data/sites_train.csv').drop(['Unnamed: 0'], axis=1)
# sites_train

X = feature_extraction(sites_train)

# split data
X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, 
                                        test_size=0.1, random_state=42)

# multi-layer perceptron classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(300,), 
    alpha=0.0001, 
    max_iter=500, 
    random_state = 42)

mlp.fit(X_train, y_train)

# prediction on thest set for hyper-parametrisatin
y_predict = mlp.predict(X_test)
print()
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# feature engineer precition data
sites_predict = seperate_sites(data_predict)

# # export seperated features
# sites_train.to_csv('../data/sites_predict.csv')

# # import seperated data again
# sites_train = pd.read_csv('../data/sites_predict.csv').drop(['Unnamed: 0'], axis=1)
# sites_train

X_predict = feature_extraction(sites_predict)

# make prediction on given data set
prediction = mlp.predict(X_predict)

# export results
result = pd.DataFrame.from_records(prediction.reshape(-1,1))
result.to_csv('../results/results.csv', index=False, header=False)
