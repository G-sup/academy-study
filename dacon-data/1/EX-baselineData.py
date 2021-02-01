import os
import glob
import random
import numpy as np
from numpy.core.fromnumeric import reshape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout,Dense,GRU,Input,Conv1D ,Flatten ,MaxPool1D,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('./z_dacon-data/train/train.csv')
submission = pd.read_csv('./z_dacon-data/sample_submission.csv')

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)

df_test = []

for i in range(81):
    file_path = './dacon-data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_pred = pd.concat(df_test)

x_train, x_test, y_train1, y_test1, y_train2, y_test2  = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], df_train.iloc[:, -1], test_size=0.3, random_state=0)
print(x_train)
print(y_train1)
print(y_train2)
print(x_pred)


x_train = x_train.values
x_test = x_test.values
y_train1 = y_train1.values
y_train2 = y_train2.values
y_test1 = y_test1.values
y_test2 = y_test2.values

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

# x = x.T.reshape(-1, 4, 5)
x_train = x_train.reshape(-1, 7, 1)
x_test = x_test.reshape(-1, 7, 1)
x_pred = x_pred.reshape(-1, 7 ,1)


from tensorflow.keras.backend import mean, maximum

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
    err=(y-pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)


for q in qunatile_list:
    model=Sequential()
    model.add(GRU(64,activation='relu',return_sequences=True, input_shape = (7,1)))
    model.add(Dropout(0.2))
    model.add(GRU(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))

    es=EarlyStopping(monitor='val_loss', mode='auto', patience=70)
    rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=35, factor=0.5)
    cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                    filepath='./dacon-data/modelcheckpoint/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    model.compile(loss=lambda x_train, y_train:quantile_loss(q, x_train, y_train), optimizer='adam')
    hist=model.fit(x_train, y_train1 ,validation_split=0.2,epochs=1000, batch_size=64, callbacks=[es, rl])
    loss=model.evaluate(x_test,y_test1)
    pred=model.predict(x_pred)
    pred = np.where(pred < 0.4, 0, pred)
    pred = np.round_(pred,3)
    y_pred=pd.DataFrame(pred)

    file_path='./dacon-data/test_test/quantile_all_7day_loss_' + str(q) + '.csv'
    y_pred.to_csv(file_path)

for q in qunatile_list:
    model=Sequential()
    model.add(GRU(64,activation='relu',return_sequences=True, input_shape = (7,1)))
    model.add(Dropout(0.2))
    model.add(GRU(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))

    es=EarlyStopping(monitor='val_loss', mode='auto', patience=70)
    rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=35, factor=0.5)
    cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                    filepath='./dacon-data/modelcheckpoint/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    model.compile(loss=lambda x_train, y_train:quantile_loss(q, x_train, y_train), optimizer='adam')
    hist=model.fit(x_train, y_train2 ,validation_split=0.2,epochs=1000, batch_size=64, callbacks=[es, rl])
    loss=model.evaluate(x_test, y_test2)
    pred=model.predict(x_pred)
    pred = np.where(pred < 0.4, 0, pred)
    pred = np.round_(pred,3)
    y_pred=pd.DataFrame(pred)

    file_path='./dacon-data/test_test/quantile_all_8day_loss_' + str(q) + '.csv'
    y_pred.to_csv(file_path)




results_1 = []

for i in range(1,10):
    file_path = './dacon-data/test_test/quantile_all_7day_loss_0.' + str(i) + '.csv'
    temp = pd.read_csv(file_path, index_col=0, header=0)
    results_1.append(temp)

results_1 = pd.concat(results_1,axis=1)
print(results_1)
#  pd.concat([df1,df2],axis=1)
results_2 = []

for i in range(1,10):
    file_path = './dacon-data/test_test/quantile_all_8day_loss_0.' + str(i) + '.csv'
    temp = pd.read_csv(file_path, index_col=0, header=0)
    results_2.append(temp)

results_2 = pd.concat(results_2,axis=1)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values


submission.to_csv('./dacon-data/submission_v3.csv', index=False)