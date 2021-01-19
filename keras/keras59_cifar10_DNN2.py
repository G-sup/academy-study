# 다차원 DNN
# (n,32,32,3) -> (n,10)
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv1D , Flatten ,MaxPool1D, LSTM, GRU, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test,  y_test) = cifar10.load_data()


from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# print(y_test.shape)  #(10000, 10)
# print(y_train.shape) #(50000, 10)

#2
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(32, 32, 3)))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()


#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k59_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=3,mode='auto')
model.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss=model.evaluate(x_test,y_test,batch_size=32)
print(loss)
