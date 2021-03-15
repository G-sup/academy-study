# 따로 사용

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping



import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try:
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
    except RuntimeError as e:
        print(e)



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

model.add(Conv2D(128, (2,2),kernel_initializer='he_normal')) 

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (2,2),kernel_regularizer=l1(l1=0.01))) 
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
