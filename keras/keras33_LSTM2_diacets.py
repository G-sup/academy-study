# 사이킷런 데이터셋 
# LSTM 으로
# Dense 와 비교

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x[:5])
# print(y[:10])
# print(x.shape)
# print(y.shape)

# print(np.max(x), np.min(y))
# print(dataset.feature_names)
# print(dataset.DESCR)


# x = x/442

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.3, random_state= 104, shuffle=True)

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape) #(92, 10)
print(x_test.shape) #(133, 10)

x = x.reshape(-1, 10, 1)
x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)
x_val = x_val.reshape(-1, 10 ,1)


#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(120, input_shape=(10,1)))
model.add(Dense(120))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='loss', patience=50,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#4
loss, mae = model.evaluate(x_test, y_test, batch_size=6)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# x_val
# loss,mae :  2761.839111328125 43.19102096557617
# RMSE :  52.55319959432148
# R2:  0.5226531608553822

# LSTM
# loss,mae :  4236.83740234375 54.57445526123047
# RMSE :  65.0909893497125
# R2:  0.26771949591976196