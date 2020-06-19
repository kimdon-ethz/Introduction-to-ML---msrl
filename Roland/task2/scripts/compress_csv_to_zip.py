import pandas as pd

df = pd.read_csv('prediction.csv')
# export results
df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')