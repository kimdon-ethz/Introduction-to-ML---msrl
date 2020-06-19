import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV 
import sklearn.metrics as metrics

def main():
    vitals = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    features_train = pd.read_csv('data/train_engineered_4.csv')
    labels_train = pd.read_csv('data/train_labels.csv')
    features_predict = pd.read_csv('data/test_engineered_4.csv')

    # set reduced_size  to reduce batch size
    reduced_size = len(features_predict)
    # reduced_size = 800

    prediction = pd.DataFrame(features_predict['pid']).iloc[0:reduced_size,:]
    metrics_summary = pd.DataFrame(columns=vitals)
    hyperparams = pd.DataFrame(columns=vitals)

    for label in vitals:
        X_train = np.array(features_train)[0:reduced_size]
        y_train = np.array(labels_train[label])[0:reduced_size]
        X_predict = np.array(features_predict)[0:reduced_size]

        # scaling data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_predict = scaler.fit_transform(X_predict)

        model = SGDRegressor(penalty='elasticnet', alpha=0.05, l1_ratio=0.1)

        print()
        print('learning : ', label)
        model.fit(X_train, y_train)

        #predict on the provided test set
        y_predicted = model.predict(X_predict)
        # y_predicted = grid.predict(X_predict)

        prediction[label] = y_predicted


    print(prediction)

    return prediction


if __name__ == "__main__":
    main()