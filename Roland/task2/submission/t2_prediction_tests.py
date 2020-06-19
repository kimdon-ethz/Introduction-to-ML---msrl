import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV 
import sklearn.metrics as metrics

def main():
    
    tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
             'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
             'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

    features_train = pd.read_csv('data/train_engineered_4.csv')
    labels_train = pd.read_csv('data/train_labels.csv')
    features_predict = pd.read_csv('data/test_engineered_4.csv')

    # set reduced_size to reduce batch size
    reduced_size = len(features_predict)
    # reduced_size = 800

    prediction = pd.DataFrame(features_predict['pid']).iloc[0:reduced_size,:]
    metrics_summary = pd.DataFrame(columns=tests)
    hyperparams = pd.DataFrame(columns=tests)

    for label in tests:
        X_train = np.array(features_train)[0:reduced_size]
        y_train = np.array(labels_train[label])[0:reduced_size]
        X_predict = np.array(features_predict)[0:reduced_size]

        # scaling data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_predict = scaler.fit_transform(X_predict)

        model = SVC(gamma='auto', kernel='rbf', C=1, probability=True, class_weight='balanced')

        print()
        print('learning : ', label)
        model.fit(X_train, y_train)

        #predict on the provided test set
        y_predicted = model.predict_proba(X_predict)
        prediction[label] = y_predicted[:,1]


    print(prediction)


    return prediction


if __name__ == "__main__":
    main()