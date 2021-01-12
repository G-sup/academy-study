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
model.add(Conv2D(32,(3,3),activation='relu',padding='same',strides=1,input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu',padding='same',strides=1))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../Data/modelCheckPoint/k46_MC_2_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=3,mode='auto')
model.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss = model.evaluate(x_test,y_test,batch_size=16)
print(loss)

# [1.3681446313858032, 0.5274999737739563]

# [1.3433111906051636, 0.5333999991416931]

# [1.2458314895629883, 0.5748999714851379]

# [1.2686151266098022, 0.6157000064849854]

# [1.14914071559906, 0.6055999994277954]