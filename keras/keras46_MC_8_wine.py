# 실습 , DNN 완성
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU , Dropout
from tensorflow.keras.callbacks import EarlyStopping
dataset = load_wine()
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data
y = dataset.target


print(x)
print(y)
print(x.shape) #(178 , 13)
print(y.shape) #(178,)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
ohe.fit(y)
y = ohe.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x , y, train_size=0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train , y_train, train_size=0.8, random_state=104)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2
model = Sequential()
model.add(Dense(128, activation='relu',input_shape=(13,)))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(27,activation='relu'))
model.add(Dense(9))
model.add(Dense(3, activation='softmax'))

#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = './modelCheckPoint/k46_MC_8_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_val,y_val) ,verbose=1,callbacks=[early_stopping,mc])

#4
loss = model.evaluate(x_test,y_test,batch_size=1)
print(loss)

# y_pred = model.predict(x_test)
# print(y_pred)
# print(y_test)

# # 결과치
# y_pred = np.argmax(y_pred,axis=-1)
# y_test = np.argmax(y_test,axis=-1)
# print(y_pred)
# print(y_test)

# [0.00092883943580091, 1.0]
# [0.02890467271208763, 0.9722222089767456]
