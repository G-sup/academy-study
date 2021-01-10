# 주말과제
# lstm (28,28,,1) 가능하지만 속도가 엄청 느리다

# 인공지능의 HELLO WORLD mnist

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout,LSTM

(x_train, y_train), (x_test,  y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)

# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape) #(28, 28)

# plt.imshow(x_train[0:2],'gray')
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.
# (x_test.reshpe(x_test.shape[0], x_test.shape[1]. x_test.shape[2], 1))

# tensorflow버전

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


# 2

model = Sequential()
model.add(LSTM(10,input_shape=(28,28)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(10,activation='softmax'))

#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=10,validation_batch_size=64,validation_split=0.2,verbose=1)

#4
loss = model.evaluate(x_test,y_test,batch_size=32)
print(loss)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(y_pred)
print(y_test)
