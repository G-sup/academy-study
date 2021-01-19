#  다차원(x) 에서 다차원(x) dnn


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
(x_train, y_train), (x_test,  y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# (x_test.reshpe(x_test.shape[0], x_test.shape[1]. x_test.shape[2], 1))

y_train = x_train
y_test = x_test


# from tensorflow.keras.utils import to_categorical
# y_test = to_categorical(y_test)
# y_train = to_categorical(y_train)



# 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(28,28,1)))
model.add(Dense(100))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))
model.summary()

#3

from tensorflow.keras.callbacks import ReduceLROnPlateau # callbacks 안에 넣어준다
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5,verbose=1) # factor = 0.5 : RL를 50%로 줄이겠다
modelpath = '../Data/modelCheckPoint/k58_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' 
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
hist = model.fit(x_train, y_train, epochs=60,batch_size=64,validation_split=0.5,verbose=1,callbacks=[early_stopping,mc,reduce_lr])

#4
loss = model.evaluate(x_test,y_test)
print(loss[0])
print(loss[1])


y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)
