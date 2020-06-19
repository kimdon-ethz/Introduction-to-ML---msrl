import pandas as pd

import t2_feature_engineering
import t2_prediction_tests
import t2_prediction_vitals

def main():

    # print(">> Feature engineering")
    # t2_feature_engineering.main()

    print(">> Predicting tests")
    results_tests = t2_prediction_tests.main()

    print(">> Predicting vitals")
    results_vitals = t2_prediction_vitals.main()

    results = pd.concat([results_tests, results_vitals.drop(['pid'], axis=1)], axis=1, sort=False)
    print(results.head())
    
    results.to_csv('prediction.csv', index=False, float_format='%.3f')
    #results_tests.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')


if __name__ == "__main__":
    main()
