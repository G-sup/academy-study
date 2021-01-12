# 인공지능의 HELLO WORLD mnist

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
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

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# (x_test.reshpe(x_test.shape[0], x_test.shape[1]. x_test.shape[2], 1))

# tensorflow버전

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


# 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout

model = Sequential()
model.add(Conv2D(128, (2,2),padding='same', strides=2, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (2,2),padding='same', strides=1))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(128, (2,2)))
model.add(Conv2D(128, (2,2)))
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

from tensorflow.keras.callbacks import ModelCheckpoint , TensorBoard # callbacks 안에 넣어준다

tb = TensorBoard(log_dir='../Data/graph', histogram_freq=0,write_graph=True,write_images=True) 

'''
 cmd 
 cd \ study
 cd \ graph
 dir/w
 tensorbord --logdir=.

 인터넷에서
 http://127.0.0.1:6006

'''

modelpath = '../Data/modelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=5,mode='auto')
hist = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc,tb])

#4
loss = model.evaluate(x_test,y_test)
print(loss[0])
print(loss[1])


# y_pred = model.predict(x_test)
# print(y_pred)
# print(y_test)

# print(y_pred[:10])
# print(y_test[:10])
# # y_test[:10]
# y_pred[:10]

# Epoch 100/1000
# 375/375 [==============================] - 32s 85ms/step - loss: 0.0169 - acc: 0.9964 - val_loss: 0.2326 - val_acc: 0.9772

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 면적을 찾아라 

plt.subplot(2,1,1) # (2,1) 짜리 1번째 그림

plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')

plt.grid() # 모눈종이형 격자 = grid


plt.title('mnist_loss n val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #

# ================

plt.subplot(2,1,2) # (2,1) 짜리 2번째 그림 shbplt = 여러장을 보려할떄

plt.plot(hist.history['acc'],marker='.',c='red',label='acc')
plt.plot(hist.history['val_acc'],marker='.',c='blue',label='val_acc')

plt.grid() 

plt.title('mnist_acc n val_acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right') 

plt.show()