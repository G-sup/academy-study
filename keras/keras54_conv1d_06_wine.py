# 실습 , DNN 완성
# 다중분류

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU,Conv1D,Flatten,MaxPool1D,Dropout
from tensorflow.keras.callbacks import EarlyStopping

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

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

print(x_train.shape) #(283, 13)
print(x_test.shape) #(152,13)


x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
x_val = x_val.reshape(-1, 13 ,1)

print(x_train.shape) #(283, 13)
print(x_test.shape) #(152,13)

#2
model = Sequential()
model.add(Conv1D(356,3, activation='relu',input_shape=(13,1)))
model.add(Dropout(0.4))
model.add(MaxPool1D(pool_size=2))
model.add(Conv1D(356,3, activation='relu',input_shape=(13,1)))
model.add(Dropout(0.4))
model.add(Dropout(0.4))
model.add(Conv1D(356,3, activation='relu',input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='VAL_loss',patience=15,mode='auto')
model.fit(x_train, y_train,epochs=500,batch_size=2,validation_data=(x_val,y_val),verbose=1,callbacks=[early_stopping])

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

# [0.00040737504605203867, 1.0]

# [0.8914732336997986, 0.9444444179534912]

# [0.3260614573955536, 0.9722222089767456]

# [0.4431506395339966, 0.9722222089767456]