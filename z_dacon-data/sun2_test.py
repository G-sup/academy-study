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


df = pd.read_csv('./z_dacon-data/test/0.csv', index_col=[0,1,2], header=0) 

print(df.info()) # [52560 rows x 5 columns]
print(df.corr()) 

print(df) #(336, 6)

df = df.values
print(df.shape)
model = load_model

model = load_model('../data/modelCheckPoint/sun__112-22127.1523.hdf5')


x_pred = df.reshape(-1,4,6)

y_pred = model.predict(x_pred)

df = pd.DataFrame(y_pred, columns=[ 'DHI','DNI','WS','RH','T','TARGET']) # header(열의 이름) 설정( 데이터는 아니다 데이터 설명용)

df.to_csv('../data/csv/sun--.csv', sep=',') 
