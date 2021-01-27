import numpy as np
from numpy.core.fromnumeric import reshape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,Conv1D ,Flatten ,MaxPool1D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

# 1
df = pd.read_csv('./z_dacon-data/train/train.csv', index_col=[0,1,2], header=0) 

df = df[['DHI','DNI','RH','T','TARGET']]


# print(df.info()) # [52560 rows x 5 columns]
# print(df.corr()) 

# print(df.iloc[-48:]) #(52560, 5)

df = df.dropna(axis=0).values


def split_x(D,x_row,y_cols):
    x , y1 = [] , []
    for i in range(len(D)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_cols+48
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number+48:y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
        x.append(tem_x)
        y1.append(tem_y)
    return np.array(x),np.array(y1)
    
x, y1 = split_x(df,4,1)

y1 = y1[:-48,:]
x = x[:-48,:]

# print('=========================================')
# ]
# print(y1[-15])
print(y1.shape)
# print('=========================================')
print(x.shape)
# print(x[-15])
# print('=========================================')

def split_x(D,x_row,y_cols):
    x1 , y2 = [] , []
    for i in range(len(D)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_cols+96 # 44
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number+96:y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
        x1.append(tem_x)
        y2.append(tem_y)
    return np.array(x1),np.array(y2)
    
x1, y2 = split_x(df,4,1)
# print(y2[-15])
print(y2.shape)

y = np.hstack((y1,y2))

df2 = pd.read_csv('./z_dacon-data/test/0.csv', index_col=[0,1,2], header=0) 
df2 = df2[['DHI','DNI','RH','T','TARGET']]

df2 = df2.dropna(axis=0).values

# print(df2)

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

x_pred = x1[-97:-1,:]

print('=========================================')

print(x_pred[17])
print(x_pred.shape)





x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
x_pred = x_pred.reshape(-1,1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 4, 5)
x_train = x_train.reshape(-1, 4, 5)
x_test = x_test.reshape(-1, 4, 5)
x_val = x_val.reshape(-1, 4, 5)
x_pred = x_pred.reshape(-1, 4 ,5)


print(x_train.shape) 
print(x_test.shape)
print(x_pred.shape)


# np.save('./z_dacon-data/sun_data.npy',arr=([x,y,x_train,x_test,x_val,y_train,y_test,y_val]))

from tensorflow.keras.backend import mean, maximum
def quantile_loss(q, y_train, y_pred):
  err = ( y_train-y_pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

# quantile = [0.1]

# for q in quantile:
#   model = Sequential()
#   model.add(GRU(256,activation='relu', input_shape = (4,5)))
#   model.add(Dropout(0.2))
#   model.add(Dense(512))
#   model.add(Dropout(0.2))
#   model.add(Dense(128))
#   model.add(Dropout(0.2))
#   model.add(Dense(2)) 
#   rd = ReduceLROnPlateau(monitor='val_loss',patience=20,factor=0.5,verbose=1)
#   es = EarlyStopping(monitor='val_loss',patience=40,mode='auto')
#   modelpath = './z_dacon-data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
#   cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')  
#   model.compile(loss=lambda y_train, y_pred: quantile_loss(q,y_train, y_pred), optimizer='adam')
#   model.fit(x_train, y_train, epochs=100, batch_size = 64, validation_data = (x_val,y_val), verbose = 1, callbacks = [es,rd,cp])
#   loss = model.evaluate(x_test,y_test,batch_size=16)
#   print(loss)
#   y_pred = model.predict(x_pred)
#   y_pred = np.where(y_pred < 0.005, 0, y_pred)
#   print(np.round_(y_pred,2))


#2
model = Sequential()
model.add(GRU(256,activation='relu', input_shape = (4,5)))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(2))
model.summary()

3
rd = ReduceLROnPlateau(monitor='val_loss',patience=20,factor=0.5,verbose=1)
es = EarlyStopping(monitor='val_loss',patience=40,mode='auto')
modelpath = './z_dacon-data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
# model.compile(loss=lambda y_train, y_pred: quantile_loss(quantile,y_train, y_pred), optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size = 16, validation_data = (x_val,y_val), verbose = 1, callbacks = [es,rd,cp])
model.save_weights("./z_dacon-data/sun__model.h5")


4
loss = model.evaluate(x_test,y_test,batch_size=16)

print(loss)
y_pred = model.predict(x_pred)

y_pred = np.where(y_pred < 0.005, 0, y_pred)

print(np.round_(y_pred,2))
