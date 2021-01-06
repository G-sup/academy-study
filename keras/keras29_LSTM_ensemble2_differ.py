# 2개의 모델 하나는 lstm 하나는 dense
# 앙상블
# 29_1 과 성능비교

import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# 1

x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])
print(x1.shape)
print(x2.shape)
print(y.shape)

# x1 = x1.reshape(13, 3, 1)
# x2 = x2.reshape(13, 3, 1)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, random_state=104)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, train_size=0.8, random_state=104)

# from sklearn.preprocessing import MinMaxScaler

# x1_predict = x1_predict.reshape(-1, 3)
# x2_predict = x2_predict.reshape(-1, 3)


# scaler = MinMaxScaler()
# scaler.fit(x1_train)
# x1_train = scaler.transform(x1_train)
# x1_test = scaler.transform(x1_test)
# x1_predict = scaler.transform(x1_predict)


# # scaler.fit(x2_train)
# x2_train = scaler.transform(x2_train)
# x2_test = scaler.transform(x2_test)
# x2_predict = scaler.transform(x2_predict)

# # print(x_train.shape) #(9, 3)
# # print(x_test.shape) #(4,3)

# # x = x.reshape(x.shape[0], x.shape[1], 1)
# x1_train = x1_train.reshape(-1, 3, 1)
# x1_test = x1_test.reshape(-1, 3, 1)
# x1 = x1.reshape(13, 3, 1)

# # x2_train = x2_train.reshape(-1, 3, 1)
# # x2_test = x2_test.reshape(-1, 3, 1)


# 2 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate

input1 = Input(shape=(3,1))
LSTM1 = LSTM(356, activation='relu')(input1)
dence1 = Dense(356)(LSTM1)
dence1 = Dense(128)(dence1)
dence1 = Dense(128)(dence1)
dence1 = Dense(128)(dence1)

input2 = Input(shape=(3,))
dence2 = Dense(356,activation='relu')(input2)
dence4 = Dense(128)(dence2)
dence4 = Dense(128)(dence4)
dence5 = Dense(128)(dence4)


merge1 = concatenate([dence1, dence5])
middlel1 = Dense(64)(merge1)
middlel2 = Dense(64)(middlel1)
middlel3 = Dense(64)(middlel1)
output1 = Dense(64)(middlel1)
output1 = Dense(32)(output1)
output1 = Dense(1)(output1)

model = Model(inputs = [input1, input2], outputs = output1)

# model.summary()

# 3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='loss',patience=60,mode='auto')
model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=4, validation_data=([x1_val,x2_val], y_val), verbose=1,callbacks=[early_stopping])

# 4
loss=model.evaluate([x1_test,x2_test],y_test, batch_size=1)
print(loss)


# print(x2_predict.shape) # 
# print(x1_predict.shape) # 
x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

y_pred = model.predict([x1_predict,x2_predict])


print(y_pred)

# 
# [17.457426071166992, 3.293602705001831]
# [[83.708786]]

# [43.21492385864258, 5.904346942901611]
# [[87.78311]]