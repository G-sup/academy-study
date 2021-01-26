import numpy as np
from numpy.core.fromnumeric import size
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
df = pd.read_csv('./z_dacon-data/train/train.csv', index_col=[0,2], header=0) 

# df2 = pd.read_csv('./z_dacon-data/test/0.csv', index_col=[0,2], header=0) 
# df2 = df2.values

print(df.info()) # [52560 rows x 7 columns]
print(df.corr()) 

# print(df.iloc[-48:]) #(52560, 7)

df = df.dropna(axis=0).values
# print(df.shape)


def split_x(D,x_row,y_cols):
    x , y = [] , []
    for i in range(len(D)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_cols+48
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number+48:y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x),np.array(y)
    
x, y = split_x(df,5,2)

print(x[63])
print('=========================================')
print(y[15])

'''
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
# x_pred = df.reshape(-1,1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 5, 7)
x_train = x_train.reshape(-1, 4, 7)
x_test = x_test.reshape(-1, 4, 7)
x_val = x_val.reshape(-1, 4, 7)
# x_pred = x_pred.reshape(-1, 4 ,7)


print(x_train.shape) 
print(x_test.shape)
# print(x_pred.shape)

# np.save('./z_dacon-data/sun_data.npy',arr=([x,y,x_train,x_test,x_val,y_train,y_test,y_val]))

# 2
model = Sequential()
model.add(GRU(256,activation='relu', input_shape = (4,7), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(7))
model.summary()

#3
rd = ReduceLROnPlateau(monitor='val_loss',patience=20,factor=0.5,verbose=1)
es = EarlyStopping(monitor='val_loss',patience=40,mode='auto')
modelpath = './z_dacon-data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size = 64, validation_data = (x_val,y_val), verbose = 1, callbacks = [es,rd,cp])
# model.save("./z_dacon-data/sun__model.h5")

#4
loss = model.evaluate(x_test,y_test,batch_size=64)
print(loss)


# y_pred = model.predict(x_pred)
# print(y_pred)
'''