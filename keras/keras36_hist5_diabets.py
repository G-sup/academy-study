# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape)
print(y.shape)

print(np.max(x), np.min(y))
print(dataset.feature_names)
print(dataset.DESCR)


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

#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(120, input_dim=10))
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
model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss',patience=10,mode='auto')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=6, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

#4
loss = model.evaluate(x_test, y_test, batch_size=6)
print('loss : ', loss)


import matplotlib.pyplot as plt


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'])
plt.show()
