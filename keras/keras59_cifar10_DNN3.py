#  다차원(x) 에서 다차원(x) dnn
# (n,32,32,3) -> (n,32,32,3)
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



x_train = x_train.reshape(-1, 32,32, 3).astype('float32')/255.
x_test = x_test.reshape(-1, 32,32, 3).astype('float32')/255.

y_train = x_train
y_test = x_test



#2
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(32, 32, 3)))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(100,activation='relu'))
model.add(Dense(3))
model.summary()


#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k59_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='mse',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=3,mode='auto')
model.fit(x_train,y_train,epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss=model.evaluate(x_test,y_test,batch_size=32)
print(loss)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)