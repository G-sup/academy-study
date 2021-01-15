import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1

x = np.load('./samsung/s2.npy',allow_pickle=True)[0]
y = np.load('./samsung/s2.npy',allow_pickle=True)[1]
x_pred = np.load('./samsung/s2.npy',allow_pickle=True)[2]
x_train = np.load('./samsung/s2.npy',allow_pickle=True)[3]
x_test = np.load('./samsung/s2.npy',allow_pickle=True)[4]
x_val = np.load('./samsung/s2.npy',allow_pickle=True)[5]
y_train = np.load('./samsung/s2.npy',allow_pickle=True)[6]
y_test = np.load('./samsung/s2.npy',allow_pickle=True)[7]
y_val = np.load('./samsung/s2.npy',allow_pickle=True)[8]


#2
model= Sequential()
model.add(GRU(512,activation='relu',input_shape=(6,1)))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#3
modelpath = '../data/modelCheckPoint/samsung_test_{epoch:02d}-{val_loss:.7f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='val_loss',patience=70,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_data=(x_val,y_val),verbose=1,callbacks=[early_stopping,mc])


#4
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss)


y_pred = model.predict(x_pred)
print(y_pred)
