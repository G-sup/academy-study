# 2 개의 파일을 만드시오
# 2.EarlyStopping 을 적용한 최고의 모델

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
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_train = scaler.transform(x_train)

# print(x_train.shape)
# print(y_train.shape)

#2 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(356,activation='relu',input_dim=13))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#3 컴파일 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
early_stoppoing = EarlyStopping(monitor='loss',patience=50,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_data=(x_val, y_val), verbose=1, callbacks=(early_stoppoing))

#4 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=4)
print('loss, mse : ',loss, mse)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# loss, mse :  11.002527236938477 2.3324036598205566
# RMSE :  3.3170058471040815
# R2:  0.8678277376002668