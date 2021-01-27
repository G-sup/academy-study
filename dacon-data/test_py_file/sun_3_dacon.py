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

x_pred = pd.concat(df_test)
print(x_pred.shape)





df_train.iloc[-48:]
print(df_train.iloc[:, -2])
x_train, x_test,y1_train,y1_test,y2_train,y2_test = train_test_split(df_train.iloc[:, :-2].values, df_train.iloc[:, -2].values,df_train.iloc[:, -1].values, test_size=0.2, random_state=104)

print(x_train.shape)
print(y1_train.shape)
print(y2_train.shape)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

# x = x.reshape(-1, 1)
# x_train = x_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)
# x_val = x_val.reshape(-1,1)
# x_pred = df.reshape(-1,1)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)

# print(x_train.shape) 
# print(x_test.shape)

# x = x.reshape(-1, 7, 1)
x_train = x_train.reshape(-1, 7, 1)
x_test = x_test.reshape(-1, 7, 1)
# x_val = x_val.reshape(-1, 6, 6)
x_pred = x_pred.reshape(-1, 7 ,1)


print(x_train.shape) 
print(x_test.shape)
print(x_pred.shape)


# 2
model = Sequential()
model.add(GRU(128,activation='relu',input_shape = (7,1)))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(1))

#3
rd = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.5,verbose=1)
es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
modelpath = '../Data/modelCheckPoint/sun__{epoch:02d}-{val_loss:.4f}.hdf5' 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train, [y1_train,y2_train], epochs=1, batch_size = 32, validation_split=0.2, verbose = 1, callbacks = [es,rd,cp])

#4
loss = model.evaluate(x_test,[y1_test,y2_test],batch_size=32)
print(loss)


y_pred = model.predict(x_pred)
print(y_pred)
