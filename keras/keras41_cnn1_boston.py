# cnn 으로구성
# 2차원을 4차원으로


import numpy as np

from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=104, shuffle=True)


# 민 맥스 스케일러 x_train

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # (506, 13)
print(x_test.shape) # (506, 13)

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

print(x_train.shape) # (506, 13)
print(x_test.shape) # (506, 13)

#2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# input1 = Input(shape=(13,1,1))
# dense1 = Conv2D(30, (2,2),padding='same', strides=2)(input1)
# dense1 = Flatten()
# dense1 = Dense(20)(dense1)
# dense1 = Dense(4)(dense1)
# output1 = Dense(1)(dense1)
# model = Model(inputs = input1, outputs = output1)
# model.summary


model = Sequential()
model.add(Conv2D(10, (2,1),padding='same', strides=2, input_shape=(13,2,1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(10))
model.summary()

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='loss',patience=20,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_split=0.2, verbose=1,callbacks=[early_stopping])

#4평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
print('loss,mae : ', loss, mae)



# 전처리후 MinMaxScailer x_train
# loss,mae :  9.900341987609863 2.169292688369751
# RMSE :  3.146480888810911
# R2:  0.8826897134778194


# cnn
# loss,mae :  25.119112014770508 3.661578416824341
