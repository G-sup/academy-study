# 사이킷런 데이터셋 
# LSTM 으로
# Dense 와 비교

import numpy as np

from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target

# print(x.shape) # (506, 13)
# print(y.shape) # (506,)
# print('===================')
# print(x[:5])
# print(y[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=104, shuffle=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, train_size=0.8, random_state=104, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# print(x_train.shape) #(283, 13)
# print(x_test.shape) #(152,13)

x = x.reshape(-1, 13, 1)
x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
x_val = x_val.reshape(-1, 13 ,1)



#2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,LSTM,Conv1D,Dropout,Flatten,MaxPool1D

input1 = Input(shape=(13,1))
dense1 = Conv1D(128,2, activation='relu')(input1)
dense1 = MaxPool1D(pool_size=2)(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Conv1D(356,2, activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Conv1D(356,2, activation='relu')(dense1)
dense1 = Flatten()(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(85)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(70)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(20)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=50,mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping])

#4평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# 전처리후 MinMaxScailer x_train
# loss,mae :  9.900341987609863 2.169292688369751
# RMSE :  3.146480888810911
# R2:  0.8826897134778194



# loss,mae :  8.75180721282959 2.1502017974853516
# RMSE :  2.958345342089606
# R2:  0.8921671253770478


# loss,mae :  13.777334213256836 2.3622894287109375
# RMSE :  3.7117834581179583
# R2:  0.8302465119160335