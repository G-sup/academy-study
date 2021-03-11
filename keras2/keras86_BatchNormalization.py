#  값을 한정시킨다. normalization, regularizer

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test,  y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(128, (2,2), strides=2, activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (2,2),kernel_initializer='he_normal')) # kernel_initializer = weight initializer(가중치 초기화) 
# (relu 계열 = he 계열), (sigmoid, tanh = Xavier 계열)이 잘먹힌다

model.add(BatchNormalization()) # 미니배치를 정규화
model.add(Activation('relu'))

model.add(Conv2D(128, (2,2),kernel_regularizer=l1(l1=0.01))) # kernel_regularizer 일반화, l1 = L1 norn(normalization)
model.add(Dropout(0.2))

model.add(Conv2D(128, (2,2),strides=2))
model.add(MaxPooling2D(2))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()

#3

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=5,mode='auto')
hist = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping])

#4
loss = model.evaluate(x_test,y_test)
print(loss[0])
print(loss[1])


# 일반화(regularizer), 정규화(normalization)= 값을 한정시킨다. 

# ====초기값 설정은 케라스 레이어의 초기 난수 가중치를 설정하는 방식=====
# kernel_initializer = weight initializer(가중치 초기화) 
# kernel_initializer = (relu 계열 = he 계열) 
#                      (sigmoid, tanh = Xavier 계열)

# bias_initializer = 
# ==================================================================
# kernel_regularizer = 일반화
#                      l1 = L1 norn(normalization)

# BatchNormalization = 미니배치를 정규화

# Dropout = 노드의 개수를 줄이는것 