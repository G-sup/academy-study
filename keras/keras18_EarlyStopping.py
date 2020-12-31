# EarlyStopping
# #3컴파일 훈련에 있음 model.fit
import numpy as np

from sklearn.datasets import load_boston

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)
print('===================')
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))     # 711.0, 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MIN MAX)
# x = x /711.
# x = (x - min) / (max - min)
#   = (x - np.min(x)) / (np.max(x) - np.min(x))

# 민 맥스 스케일러 x만
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=104, shuffle=True)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, train_size=0.8, random_state=104, shuffle=True)

# 민 맥스 스케일러 x_train

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(np.max(x), np.min(x))    # 1.0, 0.0
print(np.max(x[0]))

#2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense

input1 = Input(shape=(13,))
dense1 = Dense(120, activation='relu')(input1)
dense1 = Dense(85)(dense1)
dense1 = Dense(70)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(20)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x_train, y_train, epochs=2000, batch_size=4, callbacks=[early_stopping], validation_data=(x_val, y_val), verbose=1)

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

# 전처리 전
# loss,mae :  10.771950721740723 2.4755585193634033
# RMSE :  3.2820651926340183
# R2:  0.8723619073719358

# 전처리 후
# loss,mae :  17.4603271484375 3.0702922344207764
# RMSE :  4.1785558930421125
# R2:  0.7931105569532743

# 전처리 전
# loss,mae :  13.33849048614502 2.7177727222442627
# RMSE :  3.6521900386363155
# R2:  0.8419506790710227

#전처리 후 x/711.
# loss,mae :  13.422076225280762 2.818122625350952
# RMSE :  3.6636154345138077
# R2:  0.8409602592916448

# 전처리후 MinMaxScailer x를 통으로
# loss,mae :  10.213784217834473 2.236178159713745
# RMSE :  3.1959010670401806
# R2:  0.8789757074931831

# 전처리후 MinMaxScailer x_train
# loss,mae :  9.900341987609863 2.169292688369751
# RMSE :  3.146480888810911
# R2:  0.8826897134778194
