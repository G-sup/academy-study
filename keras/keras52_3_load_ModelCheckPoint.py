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


2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(128, (2,2),padding='same', strides=2, activation='relu', input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(128, (2,2),padding='same', strides=1))
# model.add(MaxPooling2D(pool_size=1))
# model.add(Conv2D(128, (2,2)))
# model.add(Conv2D(128, (2,2)))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10,activation='softmax'))
# model.summary()

# model.save("../../../data/h5/k52_1_model1.h5")

#3

# from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
# modelpath = '../data/modelCheckPoint/52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
# k52_1_mnist_??? => k52_1_MCK_{val_loss:.4f}.hdf5 로 이름 변경

# mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
# filepath 그지점에 W 값이 들어간다

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# early_stopping = EarlyStopping(monitor='val_loss',patience=5,mode='auto')
# hist = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping])#,mc])

# model.save("../../../data/h5/k52_1_model2.h5")
# model.save_weight("../../../data/h5/k52_1_weght.h5")


# model1 = load_model('../data/h5/k52_1_model2.h5')

#4
# loss = model1.evaluate(x_test,y_test)
# print('model1_loss : ',loss[0])
# print('model1_acc : ',loss[1])

# model.load_weights('../data/h5/k52_1_weght.h5')


# loss = model.evaluate(x_test,y_test)
# print('가중치_loss : ',loss[0])
# print('가중치_acc : ',loss[1])

# # 가중치_loss :  0.14709287881851196
# # 가중치_acc :  0.9556000232696533


# model2 = load_model('../data/h5/k52_1_model2.h5')

# result = model2.evaluate(x_test,y_test)
# print('로스 모델_loss : ',result[0])
# print('로스 모델_acc : ',result[1])

# # 로스 모델_loss :  0.14709287881851196
# # 로스 모델_acc :  0.9556000232696533

model2 = load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')

result = model2.evaluate(x_test,y_test)
print('로드 체크 포인트_loss : ',result[0])
print('로드 체크 포인트_acc : ',result[1])

# 체크 포인트_loss :  0.11576943099498749
# 체크 포인트_acc :  0.963100016117096

# 체크 포인트가 loss가 더 적다