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

for i in range(81):
    file_path = './dacon-data/test/' + str(i) + '.csv'
    df2 = pd.read_csv(file_path, index_col=[0,1,2], header=0) 
    df2 = df2[['DHI','DNI','RH','T','TARGET']]
    df2 = df2.dropna(axis=0).values
    def split_x(D,size,y_cols):
        x1 , y1 = [] , []
        for i in range(len(D)):
            x_end_number = i + size
            y_end_number = x_end_number + y_cols 
            if y_end_number > len(D) :
                break
            tem_x = D[i : x_end_number,:]
            tem_y = D[x_end_number:y_end_number, -1]  # 뒤 숫자에 따라 y가 변한다 -1 = (1개씩), : = (한 행)
            x1.append(tem_x)
            y1.append(tem_y)
        return np.array(x1),np.array(y1)
    x1, y1 = split_x(df2,4,1)
    x_pred = x1[-96:,:]
    df_test.append(x_pred)
    
X_test = np.concatenate(df_test)
X_test = X_test.reshape(-1,5)
print(X_test.shape)


# # x_pred = pd.Dataframe(X_test).add_prefix('DHI','DNI','RH','T','TARGET')
# x_pred = pd.DataFrame({'DHI': X_test[:, 0], 'DNI': X_test[:, 1], 'RH': X_test[:, 2], 'T': X_test[:, 3], 'TARGET': X_test[:, 4]})

# x_pred.to_csv('./z_dacon-data/x_pred_all.csv', sep=',') 

