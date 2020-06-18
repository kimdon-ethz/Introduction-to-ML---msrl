import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


csv = 'data/test_features.csv'
data = pd.read_csv(csv)

# list of pids in order and no redundancy
pids = pd.Series(data['pid']).drop_duplicates().tolist()
# list of feature keys
features = data.keys().values
# manual selection of non spars features
features_non_spars = ['pid', 'Time', 'Age','Temp','RRate', 'ABPm', 'ABPd', 'SpO2', 'Heartrate', 'ABPs']
features_spars = [i for i in features if i not in features_non_spars]
data_significant = data[features_non_spars]

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_features(patient_data, features = ['pid','Time','Temp','RRate', 'ABPm', 'ABPd', 'SpO2', 'Heartrate', 'ABPs']):
    """ Takes patient n and set of features. Plots the features for the given patient.
    (n refers to the n-th patient in array, not pid)
    """
    time = np.linspace(1,12,12)

    # define the figure size and grid layout properties
    figsize = (10, 8)
    cols = 3
    rows = len(features) // cols + 1
    fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axs = trim_axs(axs, len(features))

    # subplots
    i = 0
    for ax in axs:
        feature = features[i]
        ax.set_title('%s' % str(feature))
        ax.plot(time, patient_data[feature], 'o', ls='-', ms=4)
        ax.set_xlim([0,12])
        i += 1

def impute_knn_patient_data(data_patient):
    # mean of individual feature across all data
    feature_mean = data_significant.mean()
    # find features with only NaNa and replace by mean of all patients
    features_with_only_nan = data_patient.isnull().all()
    for key, value in features_with_only_nan.iteritems():
        if value == True:
            data_patient[key].loc[:] = feature_mean[key]

    # impute non-spars features
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    patient_imp = imputer.fit_transform(data_patient)
    patient_imp = pd.DataFrame(data=patient_imp, columns=data_patient.keys())

    return patient_imp

def feature_engineering_counting(data_patient):
    """ Creates a new feature data frame with:
        countin all non-Nan entries of feature"""

    # counting all non-Nan entries
    counts = data_patient.count(axis='rows')
    data_new = pd.DataFrame(counts).T
    # rename features
    for feature in data_new.keys().values:
        data_new.rename({feature: [feature+'_cnts'][0]},axis=1, inplace=True)
    return data_new


def feature_engineering_regression(data_patient):
    """ Creates a new feature data frame with: 
        linear, regression coefficienta and intercept,
        max, min value of each feature """

    # initialize new feature dataframe
    data_new = pd.DataFrame()
    for feature in data_patient.keys().values:
        if feature=='pid':
            pass
        elif feature=='Age':
            pass
        elif feature=='Time':
            pass
        else:
            reg = LinearRegression().fit(data_patient['Time'].values.reshape(-1,1), data_patient[feature])
            # append new feature columns
            data_new[feature+'_coef'] = reg.coef_
            data_new[feature+'_intrcpt'] = reg.intercept_
            data_new[feature+'_max'] = data_patient[feature].max()
            data_new[feature+'_min'] = data_patient[feature].min()

    return data_new

"""
# plot one patient data
patient_id = pids[0]
# data of only one patient
data_patient = data_significant.loc[data['pid'] == patient_id]
plot_features(data_patient, features_non_spars)
patient_imputed = impute_knn_patient_data(data_patient)
plot_features(patient_imputed, features_non_spars)
"""

# creating new data set with feature engineered data
data_new = pd.DataFrame()
print('imputing data')
# looping through every patient
for pid in pids:
    # getting data of one patient
    data_patient = data.loc[data['pid'] == pid]
    # imputing non-spars data: knn imputation
    patient_non_spars_imputed = impute_knn_patient_data(data_patient[features_non_spars])
    features_non_spars_new = feature_engineering_regression(patient_non_spars_imputed)
    # spars data: counting number of measruements
    features_spars_new = feature_engineering_counting(data_patient[features_spars])

    # new features of one patient
    age = int(data_patient.loc[:,'Age'].values[0])
    data_patient_new = pd.DataFrame({'pid': [pid], 'Age': [age]}).join(features_non_spars_new.join(features_spars_new))
    # append to new features to new data set
    data_new = data_new.append(data_patient_new)

print('exporting imputed features')
data_new.to_csv('out.csv', index=False, na_rep='nan', float_format='%.3f')

plt.show()
        
