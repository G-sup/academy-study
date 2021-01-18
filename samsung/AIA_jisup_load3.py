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

#1

x = np.load('./samsung/e-s3.npy',allow_pickle=True)[0]
y = np.load('./samsung/e-s3.npy',allow_pickle=True)[1]
x_pred = np.load('./samsung/e-s3.npy',allow_pickle=True)[2]
x_train = np.load('./samsung/e-s3.npy',allow_pickle=True)[3]
x_test = np.load('./samsung/e-s3.npy',allow_pickle=True)[4]
x_val = np.load('./samsung/e-s3.npy',allow_pickle=True)[5]
y_train = np.load('./samsung/e-s3.npy',allow_pickle=True)[6]
y_test = np.load('./samsung/e-s3.npy',allow_pickle=True)[7]
y_val = np.load('./samsung/e-s3.npy',allow_pickle=True)[8]

#1 - 2

xk = np.load('./samsung/e-k1.npy',allow_pickle=True)[0]
xk_pred = np.load('./samsung/e-k1.npy',allow_pickle=True)[1]
xk_train = np.load('./samsung/e-k1.npy',allow_pickle=True)[2]
xk_test = np.load('./samsung/e-k1.npy',allow_pickle=True)[3]
xk_val = np.load('./samsung/e-k1.npy',allow_pickle=True)[4]


model = load_model('../data/modelCheckPoint/samsung_test_3_213-3578546.50000000.hdf5')

#4
loss = model.evaluate([x_test,xk_test],y_test,batch_size=8)
print(loss)


y_pred = model.predict([x_pred,xk_pred])
print(y_pred)

