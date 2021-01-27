import numpy as np
from numpy.core.fromnumeric import reshape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,Conv1D ,Flatten ,MaxPool1D
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K
df_test = []

file_path = './dacon-data/test_test/quantile_all_loss_0.1.csv' 
df2 = pd.read_csv(file_path, index_col=[0], header=0)


df2 = df2.values
df2 = df2.reshape(96,162)
X_test = df2
print(df2)
# print(df2)

# X_test = np.concatenate(df_test)
# X_test = X_test.reshape(-1,2)
# print(X_test.shape)


# # x_pred = pd.DataFrame(X_test)
# x_pred = pd.DataFrame({'day7': X_test[:, 0],'day8': X_test[:, 1]},index = [0])

# x_pred.to_csv('./z_dacon-data/quantile_0_all.csv', sep=',')


for i in range(0,81):
    file_path = './dacon-data/test_test/quantile_all_loss_0.9.csv' 
    df2 = pd.read_csv(file_path, index_col=[0], header=0) 
    df = df2.dropna(axis=0).values
    df3 = df[96*(i-1):96*i,0]
    df4 = df[96*(i-1):96*i,1]
    # df2 = df2(96*1::,)
    df_test.append(df3)
    df_test.append(df4)
    if i == 80:
        df5 = df[-96:,0]
        df6 = df[-96:,1]
        df_test.append(df5)
        df_test.append(df6)


    
X_test = np.concatenate(df_test)
# X_test = X_test.reshape(-1,5)
print(X_test.shape)


x_pred = pd.DataFrame(X_test)
# # x_pred = pd.DataFrame({'day7': X_test[:, 0],'day8': X_test[:, 1]})

x_pred.to_csv('./dacon-data/quantile_loss/quantile_0.9_all.csv', sep=',') 
