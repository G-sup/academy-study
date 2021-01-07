# 텐서 데이터셋 
# LSTM 으로
# Dense 와 비교

import numpy as np
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state= 104)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# print(x_train.shape) #(283, 13)
# print(x_test.shape) #(152,13)


x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
x_val = x_val.reshape(-1, 13 ,1)

#2 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(356,activation='relu',input_shape=(13,1)))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#3 컴파일 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
early_stopping = EarlyStopping(monitor='loss', patience=30,mode='auto')
model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#4 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=4)
print('loss, mse : ',loss, mse)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# loss, mse :  9.877310752868652 2.2219297885894775
# RMSE :  3.1428189473448787
# R2:  0.8813448547679702



# loss, mse :  15.927762985229492 2.651428461074829
# RMSE :  3.9909599611746276
# R2:  0.8086614002725923