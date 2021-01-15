import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x = np.load('./samsung/s2.npy',allow_pickle=True)[0]
y = np.load('./samsung/s2.npy',allow_pickle=True)[1]
x_pred = np.load('./samsung/s2.npy',allow_pickle=True)[2]
x_train = np.load('./samsung/s2.npy',allow_pickle=True)[3]
x_test = np.load('./samsung/s2.npy',allow_pickle=True)[4]
x_val = np.load('./samsung/s2.npy',allow_pickle=True)[5]
y_train = np.load('./samsung/s2.npy',allow_pickle=True)[6]
y_test = np.load('./samsung/s2.npy',allow_pickle=True)[7]
y_val = np.load('./samsung/s2.npy',allow_pickle=True)[8]

#3
model = load_model('../data/modelCheckPoint/samsung_test_3_01-2635745024.0000000.hdf5')

result = model.evaluate(x_test,y_test)
print('로드 체크 포인트_loss : ',result[0])
print('로드 체크 포인트_mse : ',result[1])

y_pred = model.predict(x_pred)
print(y_pred)
