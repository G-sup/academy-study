# SimpleRNN

import numpy as np

#1 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1)

#2 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1)))  # default activation = linear for SimpleRNN
# Param 의 개수 = 노드의 개수 * ( 노드의 개수 + input_dim + bios(1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
model.summary()

'''
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

# 0.000298828148515895
# [[8.0292635]]
'''