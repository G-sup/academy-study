import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU,Input
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob  


x = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[0]
y = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[1]
x_train = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[2]
x_test = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[3]
x_val = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[4]
y_train = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[5]
y_test = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[6]
y_val = np.load('./z_dacon-data/sun_data.npy',allow_pickle=True)[7]

df2 = pd.read_csv('./z_dacon-data/test/0.csv', index_col=[0,2], header=0) 

df2 = df2.dropna(axis=0).values
print(df2)

def split_x(D,size,y_cols):
    x1 , y1 = [] , []
    for i in range(len(D)):
        x_end_number = i + size
        y_end_number = x_end_number + y_cols 
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number:y_end_number, :]  # 뒤 숫자에 따라 y가 변한다 -1 = (1개씩), : = (한 행)
        x1.append(tem_x)
        y1.append(tem_y)
    return np.array(x1),np.array(y1)
    
x1, y1 = split_x(df2,4,1)

# print(x)
print('=============================================')
x_pred = x1[-96:,:]
print(x_pred)
print(x_pred.shape)


model = load_model('./z_dacon-data/modelCheckPoint/sun__66-19665.0156.hdf5')


y_pred = model.predict(x_pred)
print(y_pred)
