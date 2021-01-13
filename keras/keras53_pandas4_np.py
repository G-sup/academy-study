import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) 

print(df)

print(df.shape) # (150, 5)
print(df.info())

# numpy로 저장하는법

# x = df.to_numpy()
x = df.values

print(x)
print(type(x))

np.save('../data/npy/iris_sklearn.npy', arr=x)

