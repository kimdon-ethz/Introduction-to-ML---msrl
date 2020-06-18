#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.1"
__maintainer__ = "Jonas Lussi"
__email__ = "jlussi@ethz.ch"

"""Brief: Trains a MLP predictor to classify Protein strings
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


# --Load Train and Test Data--
df_train = pd.read_csv(os.path.join('..','data','train.csv'))
data_X=df_train['Sequence']
data_y=df_train['Active']

df_test = pd.read_csv(os.path.join('..','data','test.csv'))
#--------------------

#--------------- Check Class distribution ---------------
print('Inactive (0) and Active (1):')
print(data_y.value_counts())
print('%f%% of all proteins are active' % \
     (data_y.value_counts()[1]/
     data_y.value_counts()[0]*100))
#--------------------------------------------------------

#--------- Data Reduction ------
# # reduce data size for faster prototyping
#df_train = df_train.iloc[0:500,:]
#df_test=df_test.iloc[0:500,:]
#-------------------------------

#--------- Split data up in test and evaluation  ------
# needs to be done now, so we don't affect the evaluation data when manipulating the data

df_train, df_evaluate = train_test_split(df_train,
                                        test_size=0.1, random_state=123)
#------------------------------------------------------

#--------------- Resampling Data for better results ---------------
# Separate majority and minority classes
inactive_majority = df_train[df_train.Active==0]
active_minority = df_train[df_train.Active==1]

# Upsample minority class to total of 1/2 of majority class
minority_upsampled = resample(active_minority,
                                 replace=True,
                                 n_samples=int(inactive_majority.Active.value_counts()[0]/1.5),
                                 random_state=47)

# Downsample majority class
majority_downsampled = resample(inactive_majority,
                                 replace=True,
                                 n_samples=int(inactive_majority.Active.value_counts()[0]/1.5),
                                 random_state=47)

#undoing up/downsampling, because its not required (after hyperparameter optimization):
minority_upsampled=active_minority
majority_downsampled=inactive_majority
# Combine majority class with upsampled minority class
df_train_clean = pd.concat([majority_downsampled, minority_upsampled])
# shuffle rows
df_train_clean = df_train_clean.sample(frac=1,random_state=42)
print(df_train_clean.Active.value_counts())
#------------------------------------------------------------------


#---------------- Encoding ----------------------
# Create one single Label encoder to keep track of label-mapping
label_encoder = LabelEncoder()
def encoding(df_train_clean):
    #Create empty lists
    integer_encoded = [[],[],[],[]]
    onehot_encoded= [[],[],[],[]]

    # Create integer Encoding and then oneHot Encoding for each Char seperately:

    for i in range(0,4):
        #loop through each char
        integer_encoded[i] = label_encoder.fit_transform(df_train_clean.Sequence.str[i])

        # change integer encoding to oneHot encoding
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded[i] = integer_encoded[i].reshape(len(integer_encoded[i]), 1)
        onehot_encoded[i] = onehot_encoder.fit_transform(integer_encoded[i])


    train_data_X_encoded=np.concatenate((onehot_encoded[0],onehot_encoded[1],onehot_encoded[2],onehot_encoded[3]),axis=-1)
    return train_data_X_encoded
train_data_X_encoded=encoding(df_train_clean)
evaluate_data_X_encoded=encoding(df_evaluate)
#-------------------------------------------------

print(train_data_X_encoded.shape)
#----------- Create Classifier ------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(500,),
    alpha=0.0001,
    max_iter=23,
    random_state = 42,
    verbose=True,
    activation='relu',
    solver='adam',
    batch_size=100)
#print to get current settings to evaluate all started models
print(mlp)

#-----------------------------------------------

#----------- Finding Hyperparmeters for neural net (classifier) ------
# seperate features from labels for both evaulation and test set
y = df_train_clean.drop(['Sequence'], axis=1).values.ravel()
y_eval=df_evaluate.drop(['Sequence'], axis=1).values.ravel()


mlp.fit(train_data_X_encoded, y)
y_train_predict = mlp.predict(train_data_X_encoded)
y_predict = mlp.predict(evaluate_data_X_encoded)
# compare f1 scores to see if system is overfitting
print("f1_score on evaluation set:")
print(f1_score(y_eval, y_predict))
print("f1_score on test set:")
print(f1_score(y, y_train_predict))
print(confusion_matrix(y_eval, y_predict))
print(classification_report(y_eval, y_predict))
#-------------------------------------------------------------------

#
#---------------- Predicting -------------------
# Predict on Test Data
# first we need to fit  on whole data again:
mlp.fit(np.concatenate((train_data_X_encoded,evaluate_data_X_encoded)),np.concatenate((y,y_eval)))
# Encode test data
test_X_encoded=encoding(df_test)
# predict on test data
prediction = mlp.predict(test_X_encoded)
#-----------------------------------------------



#---------------- Exporting -------------------
result = pd.DataFrame.from_records(prediction.reshape(-1,1))
result.to_csv('../results/out_Task3.csv', header= False, index=False)
#----------------------------------------------


