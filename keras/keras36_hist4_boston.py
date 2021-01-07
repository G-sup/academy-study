# x_val을 넣어보면 성능이 더 좋아진다( 안좋아 지는 경우도 있지만 데이터가 많을수록 더 좋아진다 )


import numpy as np

from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import EarlyStopping

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
dense1 = Dense(30)(dense1)
dense1 = Dense(20)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss',patience=20,mode='auto')
hist= model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping])

#4평가 예측
loss = model.evaluate(x_test, y_test, batch_size=4)
print('loss: ', loss)

import matplotlib.pyplot as plt


# plt.plot(x, y) 와  plt.show() 만 해도 돌아간다
# (x,y)에서 한항목만 넣으면 y자리에 넣어서 순서대로 나온다

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'])
plt.show()
