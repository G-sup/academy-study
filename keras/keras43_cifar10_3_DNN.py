import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , Flatten ,MaxPool2D, LSTM, GRU, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

(x_train, y_train), (x_test,  y_test) = cifar10.load_data()


# print(x_test.shape) #(10000, 32, 32, 3)
# print(x_train.shape) #(50000, 32, 32, 3)
# print(y_test.shape)  #(10000, 1)
# print(y_train.shape) #(50000, 1)
 

# plt.imshow(x_train[0],'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_test = x_test.reshape(-1,3072)
x_train = x_train.reshape(-1,3072)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# print(y_test.shape)  #(10000, 10)
# print(y_train.shape) #(50000, 10)

#2
model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(3072,)))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='loss',patience=3,mode='auto')
model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.2,verbose=1,callbacks=[early_stopping])

#4
loss=model.evaluate(x_test,y_test,batch_size=32)
print(loss)


# [2.3039894104003906, 0.10000000149011612]