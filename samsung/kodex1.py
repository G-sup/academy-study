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

df = pd.read_csv('../study/samsung/KODEX.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 

a = df[['종가','고가','저가','금액(백만)','거래량','시가']]


a.columns = ['close','high','low','price(million)', 'volume','start'] # 열(columns)의 이름변경


print(a)
print(a.info())

a = a.loc[::-1]

print(a)
print(a.tail()) # 디폴트 5 = df[-5:]
print(a.info())
print(a.describe())

print(a)
print(a.info())

x = a.dropna(axis=0).values

print(x)


x_pred = x[1081 : 1087,:]
print('x_pred')

print(x_pred)


def split_x(seq,size,cols):
    x , y = [] , []
    for i in range(len(seq)):
        x_end_number = i + size
        y_end_number = x_end_number + cols -1
        if y_end_number > len(seq) :
            break
        tem_x = seq[i : x_end_number]
        tem_y = seq[x_end_number-1:y_end_number , -1]
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x),np.array(y)
    
x, y = split_x(x,6,1)


print(y.shape)
print(x_pred.shape)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
x_pred = x_pred.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 6, 6)
x_train = x_train.reshape(-1, 6, 6)
x_test = x_test.reshape(-1, 6, 6)
x_val = x_val.reshape(-1, 6 ,6)


print(x_train.shape) 
print(x_test.shape)

#2
model= Sequential()
model.add(GRU(512,activation='relu',input_shape=(6,6)))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.summary()

#3
modelpath = '../data/modelCheckPoint/samsung_test_3_{epoch:02d}-{val_loss:.7f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='val_loss',patience=70,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_data=(x_val,y_val),verbose=1,callbacks=[early_stopping,mc])


#4
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss)


y_pred = model.predict(x_pred)
print(y_pred)
