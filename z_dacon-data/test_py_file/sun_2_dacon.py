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
import pandas as pd
import numpy as np
import os
import glob
import random



# 1
train = pd.read_csv('./z_dacon-data/train/train.csv', index_col=[0,1,2], header=0) 


def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[[ 'TARGET', 'DHI', 'DNI', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[[ 'TARGET', 'DHI', 'DNI', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
print(df_train.iloc[:48])

# train.iloc[48:96]
# train.iloc[48+48:96+48]
# print(df_train.tail())

df_test = []

for i in range(81):
    file_path = './z_dacon-data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
X_test.shape





df_train.iloc[-48:]
print(df_train.iloc[:, -2])


X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
    # (a) Modeling  
    model = Sequential(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model

# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
# results_1.sort_index()[:48]



# # 2
# model = Sequential()
# model.add(GRU(128,activation='relu',input_shape = (7,1)))
# model.add(Dropout(0.3))
# model.add(Dense(512))
# model.add(Dropout(0.3))
# model.add(Dense(512))
# model.add(Dropout(0.3))
# model.add(Dense(256))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Dense(1))

# #3
# rd = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.5,verbose=1)
# es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
# modelpath = '../Data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
# cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
# model.compile(loss='mse',optimizer='adam',metrics=['mae'])
# model.fit(x_train, [y1_train,y2_train], epochs=1, batch_size = 32, validation_split=0.2, verbose = 1, callbacks = [es,rd,cp])

# #4
# loss = model.evaluate(x_test,[y1_test,y2_test],batch_size=32)
# print(loss)


# y_pred = model.predict(x_pred)
# print(y_pred)
