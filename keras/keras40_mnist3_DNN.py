# 주말과제
# DNN 로구성 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test,  y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)

# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape) #(28, 28)

# plt.imshow(x_train[0:2],'gray')
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.
# (x_test.reshpe(x_test.shape[0], x_test.shape[1]. x_test.shape[2], 1))

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y_test = y_test.reshape(-1,1)
# ohe.fit(y_test)
# y_test = ohe.transform(y_test).toarray()

# y_train = y_train.reshape(-1,1)
# ohe.fit(y_train)
# y_train = ohe.transform(y_train).toarray()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)

print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)


#2
model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(34))
model.add(Dense(34))
model.add(Dense(1,activation='softmax'))

#3
model.compile(loss='categorical_clossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)

#4
loss= model.evaluate(x_test,y_test)
print(loss)


y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

print(y_pred[:10])
print(y_test[:10])
