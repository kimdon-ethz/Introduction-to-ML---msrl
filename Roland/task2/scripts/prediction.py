import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.svm import SVC

tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
vitals = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

features_train = pd.read_csv('data/train_engineered_4.csv')
labels_train = pd.read_csv('data/train_labels.csv')
features_test = pd.read_csv('data/test_engineered_4.csv')

tests = ['LABEL_Sepsis']
prediction = pd.DataFrame(features_test['pid'])
for label in tests:
    X_train = np.array(features_train)
    y_train = np.array(labels_train[label])
    X_test = np.array(features_test)
    # scale data
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    clf = SVC(gamma='auto', kernel='sigmoid', probability=True, class_weight='balanced')

    print('learning : ', label)
    clf.fit(X_train, y_train)

    print('predicting : ', label)
    predicted_label = clf.predict_proba(X_test)
    prediction[label] = predicted_label[:,1]

prediction.to_csv('prediction.csv', index=False, float_format='%.3f')