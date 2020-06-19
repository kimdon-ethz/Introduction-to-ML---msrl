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

    #vitals = ['LABEL_Sepsis']

    # set reduced_size  to reduce batch size
    reduced_size = len(features_predict)
    #reduced_size = 800

    prediction = pd.DataFrame(features_predict['pid']).iloc[0:reduced_size,:]
    metrics_summary = pd.DataFrame(columns=vitals)
    hyperparams = pd.DataFrame(columns=vitals)

    for label in vitals:
        X_train = np.array(features_train)[0:reduced_size]
        y_train = np.array(labels_train[label])[0:reduced_size]
        X_predict = np.array(features_predict)[0:reduced_size]

        # split into test and validation for hyperparameter tuning
        X_train, X_test, y_train, y_test = train_test_split( 
                            X_train, y_train, 
                            test_size = 0.20, random_state = 101)

        print()
        print('X_train: ', np.shape(X_train))
        print('X_test: ', np.shape(X_test))
        print('y_train: ', np.shape(y_train))
        print('y_test: ', np.shape(y_test))

        # scaling data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_predict = scaler.fit_transform(X_predict)

        model = SGDRegressor(penalty='elasticnet', alpha=0.05, l1_ratio=0.1)

        print()
        print('learning : ', label)
        model.fit(X_train, y_train)

        print('predicting : ', label)
        predicted_label = model.predict(X_test)
        metrics_default = metrics.r2_score(np.array(y_test), np.array(predicted_label))
        print('metrics ROC : ', label)
        print(metrics_default)

        ################# 
        # defining parameter range 
        param_grid = {#'l1_ratio': [0.01, 0.1, 1],
                      'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      'penalty': ['elasticnet']
                      }
        
        grid = GridSearchCV(SGDRegressor(), param_grid, refit = True, verbose = 3, scoring='r2') 
          
        # fitting the model for grid search 
        grid.fit(X_train, y_train)

        print()
        print('best parameter after tuning : ')
        print(grid.best_params_) 
        print()
        print('model after hyper-parameter tuning ')
        print(grid.best_estimator_)
        hyperparams[label] = [grid.best_estimator_.l1_ratio, 
                              grid.best_estimator_.alpha, 
                              grid.best_estimator_.penalty]
        
        # predicting on the test set to get a notion of the metrics
        grid_predictions = grid.predict(X_test) 
        # print classification report 
        metrics_grid = metrics.r2_score(np.array(y_test), np.array(grid_predictions))
        print()
        print('metrics ROC : ', label)
        print(metrics_grid)
        metrics_summary[label] = [metrics_grid]
        #################

        #predict on the provided test set
        y_predicted = model.predict(X_predict)
        prediction[label] = y_predicted

    print()
    print('hyperparams:')
    print(hyperparams)

    print()
    print('metrics of all labels:')
    print(metrics_summary)
    print(metrics_summary.mean(axis=1))
        # print(predicted_label)
        # print(np.shape(prediction))
        # prediction[label] = predicted_label

    print(prediction)
    #prediction.to_csv('prediction_vitals.csv', index=False, float_format='%.3f')

    return prediction


if __name__ == "__main__":
    main()