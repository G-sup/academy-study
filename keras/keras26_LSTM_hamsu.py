
# LSTM = 3차원 (행(1),과 열(2), 몇개씩 자를것인지(3)) = ( , , , )

import numpy as np

#1 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1)


#2 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

inputs = Input(shape=(3,1))
dense1 = (LSTM(356, activation='relu'))(inputs)
dense1 = (Dense(128))(dense1)
dense1 = (Dense(128))(dense1)
dense1 = (Dense(64))(dense1)
dense1 = (Dense(64))(dense1)
dense1 = (Dense(64))(dense1)
outputs = (Dense(1))(dense1)
model = Model(inputs = inputs, outputs= outputs)
# model.summary()

#3
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4
loss = model.evaluate(x ,y)
print(loss)

x_pred = np.array([5,6,7]) #(3,)
x_pred = x_pred.reshape(1, 3, 1)
# LSTM과 형태를 맞춰줘야 하기 때문에 프레딕트도 형태를 변화 시킨다

result = model.predict(x_pred)
print(result)

# 0.0015123276971280575
# [[8.027143]]