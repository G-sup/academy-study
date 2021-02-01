import numpy as np
from numpy.core.fromnumeric import reshape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,Conv1D ,Flatten ,MaxPool1D
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K
# 1
df = pd.read_csv('./dacon-data/train/train.csv', index_col=[0,1,2], header=0) 

df = df[['DHI','DNI','RH','T','TARGET']]


# print(df.info()) # [52560 rows x 5 columns]
# print(df.corr()) 

# print(df.iloc[-48:]) #(52560, 5)

df = df.dropna(axis=0).values


def split_x(D,x_row,y_cols):
    x , y1 = [] , []
    for i in range(len(D)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_cols+96
        if y_end_number > len(D) :
            break
        tem_x = D[i : x_end_number,:]
        tem_y = D[x_end_number+96:y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
        x.append(tem_x)
        y1.append(tem_y)
    return np.array(x),np.array(y1)
    
x, y1 = split_x(df,4,1)

y = y1[:-48,:]
x = x[:-48,:]

# print('=========================================')
# ]
# print(y1[-15])
print(y)
# print('=========================================')
print(x)
# print(x[-15])
# print('=========================================')

# def split_x(D,x_row,y_cols):
#     x1 , y2 = [] , []
#     for i in range(len(D)):
#         x_end_number = i + x_row
#         y_end_number = x_end_number + y_cols+96 # 44
#         if y_end_number > len(D) :
#             break
#         tem_x = D[i : x_end_number,:]
#         tem_y = D[x_end_number+96:y_end_number, -1] # 뒤 숫자에 따라 y가 변한다
#         x1.append(tem_x)
#         y2.append(tem_y)
#     return np.array(x1),np.array(y2)
    
# x1, y2 = split_x(df,4,1)
# # print(y2[-15])
# print(y2.shape)

# y = np.hstack((y1,y2))
# print(y.shape)
# y = y1

df2 = pd.read_csv('./dacon-data/x_pred_all.csv', header=0) 
df2 = df2[['DHI','DNI','RH','T','TARGET']]

df2 = df2.dropna(axis=0).values

print(df2.shape)

x_pred = df2.reshape(7776, 4, 5)

# def split_x(D,size,y_cols):
#     x1 , y1 = [] , []
#     for i in range(len(D)):
#         x_end_number = i + size
#         y_end_number = x_end_number + y_cols 
#         if y_end_number > len(D) :
#             break
#         tem_x = D[i : x_end_number,:]
#         tem_y = D[x_end_number:y_end_number, -1]  # 뒤 숫자에 따라 y가 변한다 -1 = (1개씩), : = (한 행)
#         x1.append(tem_x)
#         y1.append(tem_y)
#     return np.array(x1),np.array(y1)
    
# x1, y1 = split_x(df2,4,1)

# # x_pred = x1[-97:-1,:]
# x_pred = x1

print(x)
print(x_pred)
print(x_pred.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)




x_train = x_train.reshape(-1, 5)
x_test = x_test.reshape(-1, 5)
x_val = x_val.reshape(-1,5)
x_pred = x_pred.reshape(-1,5)

print(x_train) 

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)
# print(x_train) 

print(x_train.shape) 
print(x_test.shape)

# x = x.T.reshape(-1, 4, 5)
x_train = x_train.reshape(-1, 4, 5)
x_test = x_test.reshape(-1, 4, 5)
x_val = x_val.reshape(-1, 4, 5)
x_pred = x_pred.reshape(-1, 4 ,5)


print(x_train) 
# print(x_test)
# print(x_pred.shape)

# print(y_train) 
# print(y_test)

# np.save('./z_dacon-data/sun_data.npy',arr=([x,y,x_train,x_test,x_val,y_train,y_test,y_val]))

from tensorflow.keras.backend import mean, maximum

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
    err=(y-pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

for q in qunatile_list:
    model=Sequential()
    model.add(Dense(256,activation='relu', input_shape = (4,5)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))

    es=EarlyStopping(monitor='val_loss', mode='auto', patience=50)
    rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=25, factor=0.5)
    cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                    filepath='./dacon-data/modelcheckpoint/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    model.compile(loss=lambda x_train, y_train:quantile_loss(q, x_train, y_train), optimizer='adam')
    hist=model.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=1, batch_size=16, callbacks=[es, rl])
    loss=model.evaluate(x_test, y_test)
    pred=model.predict(x_pred)
    pred = np.where(pred < 0.4, 0, pred)
    pred = np.round_(pred,3)
    pred = pred.reshape(-1,1)
    y_pred=pd.DataFrame(pred)

    file_path='./dacon-data/test_test/quantile_all_loss_Dense' + str(q) + '.csv'
    y_pred.to_csv(file_path)

#2
# model = Sequential()
# model.add(GRU(256,activation='relu', input_shape = (4,5)))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.summary()

# 3
# rd = ReduceLROnPlateau(monitor='val_loss',patience=20,factor=0.5,verbose=1)
# es = EarlyStopping(monitor='val_loss',patience=40,mode='auto')
# modelpath = './z_dacon-data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
# cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
# model.compile(loss='mse',optimizer='adam',metrics=['mae'])
# # model.compile(loss=lambda y_train, y_pred: quantile_loss(quantile,y_train, y_pred), optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size = 16, validation_data = (x_val,y_val), verbose = 1, callbacks = [es,rd,cp])
# model.save_weights("./z_dacon-data/sun__model.h5")

# model = load_model('./z_dacon-data/modelCheckPoint/sun__94-132.4603.hdf5')


# 4
# loss = model.evaluate(x_test,y_test,batch_size=16)

# print(loss)
# y_pred = model.predict(x_pred)

# y_pred = np.where(y_pred < 0.5, 0, y_pred)

# print(np.round_(y_pred,2))
