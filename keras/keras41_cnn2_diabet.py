# cnn 으로구성
# 2차원을 4차원으로

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) #(442,10)
print(y.shape) #(442,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_test = x_test.reshape(-1,10,1,1)
x_train = x_train.reshape(-1,10,1,1)

#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(10, (2,1) , input_shape = (10,1,1)))
# model.add(Conv2D(10, (2,1),padding='same', strides=2, input_shape=(13,2,1)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(1))
model.summary()
#3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=6, validation_split=0.2, verbose=1, callbacks=[early_stopping])

#4
loss, mae = model.evaluate(x_test, y_test, batch_size=6)
print('loss,mae : ', loss, mae)


# x_val
# loss,mae :  2761.839111328125 43.19102096557617
# RMSE :  52.55319959432148
# R2:  0.5226531608553822


# cnn
# loss,mae :  2575.578369140625 41.10071563720703