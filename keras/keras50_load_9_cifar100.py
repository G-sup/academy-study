import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , Flatten ,MaxPool2D, LSTM, GRU, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

x_train = np.load('../data/npy/cifar100_x_train.npy') 
x_test = np.load('../data/npy/cifar100_x_test.npy') 
y_train = np.load('../data/npy/cifar100_y_train.npy') 
y_test = np.load('../data/npy/cifar100_y_test.npy') 

# print(x_test.shape) #(10000, 32, 32, 3)
# print(x_train.shape) #(50000, 32, 32, 3)
# print(y_test.shape)  #(10000, 1)
# print(y_train.shape) #(50000, 1)
 

# plt.imshow(x_train[0],'gray')
# # plt.imshow(x_train[0])
# plt.show()

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# print(y_test.shape)  #(10000, 10)
# print(y_train.shape) #(50000, 10)

#2
model = Sequential()
model.add(Conv2D(30,(2,2),padding='same',strides=2,input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(50,(2,2),padding='same',strides=1))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(50,(2,2),padding='same',strides=1))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(128,(2,2),padding='same',strides=1))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(100,activation='softmax'))
model.summary()

#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k46_MC_3_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=3,mode='auto')
model.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss=model.evaluate(x_test,y_test,batch_size=32)
print(loss)

# [2.9148507118225098, 0.28279998898506165]