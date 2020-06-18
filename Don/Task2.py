import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score, r2_score

# import the data file
df = pd.read_csv('train_features.csv')
dy = pd.read_csv('train_labels.csv')
dt = pd.read_csv('test_features.csv')

#---Compare the number of missing values in each column---
#missing = df.isnull().sum(0).reset_index()
#missing.columns = ['column', 'count']
#missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
#missing['percentage'] = missing['count'] / float(df.shape[0]) * 100
#ind = np.arange(missing.shape[0])
#width = 1
#fig, ax = plt.subplots(figsize=(8,10))
#rects = ax.barh(ind, missing.percentage.values, color='b')
#ax.set_yticks(ind)
#ax.set_yticklabels(missing.column.values, rotation='horizontal')
#ax.set_xlabel("Precentage of missing values %", fontsize = 10)
#ax.set_title("Number of missing values in each column", fontsize = 12)
#plt.show()
#--------------------------------------------------------

#-----------------Correlation diagram--------------------
#colormap = plt.cm.RdBu
#plt.figure(figsize=(24,22))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
#sns.heatmap(X_trains.astype(float).corr(),linewidths=0.1, vmax=1.0, \
#            square=True, cmap=colormap, linecolor='white', annot=True)
#--------------------------------------------------------

# assign X and y from the train dataset
all_pids = df.groupby('pid')
ap = list(all_pids.groups.keys())
train_pids, val_pids = train_test_split(ap, test_size = 0.2, \
                                        random_state = 42)

vital = ['Heartrate', 'SpO2', 'ABPs', 'ABPm', 'RRate','ABPd']
test = ['Temp', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess', \
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', \
         'FiO2', 'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', \
         'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', \
         'Hct', 'Bilirubin_total', 'TroponinI', 'pH']
tl = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', \
      'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', \
      'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']
vl = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

#X_train = pd.DataFrame()
#X_val = pd.DataFrame()
#Y_train = pd.DataFrame()
#Y_val = pd.DataFrame()
#itt = IterativeImputer(max_iter=20, tol=0.01, random_state = 42)
#preprocessor = ColumnTransformer(transformers = [ \
#                                        ('vital', itt, vital)])
#sc = StandardScaler()
#
##Imputing missing values
#X_i = df.copy()
#X_i.loc[:,test] = X_i.loc[:,test].notnull().astype('float')
#imp = preprocessor.fit(X_i)
#X_i.loc[:,vital] = imp.transform(X_i)
#X_i = X_i.drop(['Bilirubin_total', 'PaCO2', 'BaseExcess', 'Alkalinephos', \
#             'HCO3', 'PTT', 'Phosphate', 'Magnesium', 'Creatinine', \
#             'Calcium', 'Platelets', 'WBC'], axis = 1)
#
#for pid in train_pids:
#    X_pid = X_i.groupby('pid').get_group(pid)
#    X_pid = X_pid.drop(['pid', 'Time'], axis=1)
#    xp = X_pid.iloc[11].values
#    xi = X_pid.iloc[:,:].drop(['Age'], axis=1).values.flatten()
#    xj = pd.DataFrame([[*xi, *xp]])
#    X_train = X_train.append(xj)
#X_trains = pd.DataFrame(sc.fit_transform(X_train))
#
#for pid in val_pids:
#    X_pid = X_i.groupby('pid').get_group(pid)
#    X_pid = X_pid.drop(['pid', 'Time'], axis=1)
#    xp = X_pid.iloc[11].values
#    xi = X_pid.iloc[:,:].drop(['Age'], axis=1).values.flatten()
#    xj = pd.DataFrame([[*xi, *xp]])
#    X_val = X_val.append(xj)
#X_vals = pd.DataFrame(sc.fit_transform(X_val))
#
#
##Split y values into train and validation set
#ypids = dy.groupby('pid')
#for pid in train_pids:
#    Y_pid = ypids.get_group(pid)
#    Y_pid = Y_pid.drop(['pid'], axis=1)
#    Y_train = Y_train.append(Y_pid)
#
#for pid in val_pids:
#    Y_pid = ypids.get_group(pid)
#    Y_pid = Y_pid.drop(['pid'], axis=1)
#    Y_val = Y_val.append(Y_pid)
#    
##Test set
#X_t = dt.copy()
#X_t.loc[:,test] = X_t.loc[:,test].notnull().astype('float')
#X_t.loc[:,vital] = imp.transform(X_t)
#X_t = X_t.drop(['Bilirubin_total', 'PaCO2', 'BaseExcess', 'Alkalinephos', \
#             'HCO3', 'PTT', 'Phosphate', 'Magnesium', 'Creatinine', \
#             'Calcium', 'Platelets', 'WBC'], axis = 1)
#
#test_pid = list(dt.groupby('pid').groups.keys())
#X_test = pd.DataFrame()
#for pid in test_pid:
#    X_pid = X_t.groupby('pid').get_group(pid)
#    X_pid = X_pid.drop(['pid', 'Time'], axis=1)
#    xp = X_pid.iloc[11].values
#    xi = X_pid.iloc[:,:].drop(['Age'], axis=1).values.flatten()
#    xj = pd.DataFrame([[*xi, *xp]])
#    X_test = X_test.append(xj)   
#X_tests = pd.DataFrame(sc.fit_transform(X_test))


svc = SVC(kernel='rbf', C=0.1, class_weight='balanced', gamma='scale', \
          probability=True, random_state=0)
svr = SGDRegressor(penalty='elasticnet', alpha=0.05, l1_ratio=0.1)
Y_predd = pd.DataFrame()
Y_predd = Y_predd.append([pd.DataFrame(test_pid, columns=['pid'])])

#for i in ['LABEL_Sepsis']:
#    svc.fit(X_trains, Y_train[i])
#    y_pred = svc.predict(X_vals)
#    print('roc_auc_score:', roc_auc_score(Y_val[i], y_pred))
    
for i in tl:
    svc.fit(X_trains, Y_train[i])
    y_pred = svc.predict_proba(X_tests)
    Y_predd[i] = pd.DataFrame(y_pred)[1]

#for i in vl:
#    svr.fit(X_trains, Y_train[i])
#    y_pred = svr.predict(X_vals)
#    print('r2 score', r2_score(Y_val[i], y_pred))

for i in vl:
    svr.fit(X_trains, Y_train[i])
    y_pred = svr.predict(X_tests)
    Y_predd[i] = pd.DataFrame(y_pred)

Y_predd.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
    
