import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split

# 1
x = np.array([[1,2,3], [2,3,4,], [3,4,5], [5,6,7], [4,5,6], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

x = x.reshape(13, 3, 1)


# print(x.shape) #(13, 3)
# print(y.shape) #(13,)
# print(x_pred.shape) #(13, 3)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=104)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=104)


# 3차원일때  MinMaxScaler를 사용할수 없기떄문에 2차원으로 만들고 스케일링후 다시 3차원으로 만든다음에 돌린다

# from sklearn.preprocessing import MinMaxScaler
# x_pred = x_pred.reshape(1, 3)
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_pred = scaler.transform(x_pred)

# print(x_train.shape) #(9, 3)
# print(x_test.shape) #(4,3)

# x = x.reshape(13, 3, 1)
# x_train = x_train.reshape(9, 3, 1)
# x_test = x_test.reshape(4, 3, 1)


#2

model = Sequential()
model.add(Conv1D(128, 2, input_shape=(3,1),activation='relu'))
model.add(Dropout(0.4))
model.add(Conv1D(356, 2, input_shape=(3,1),activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#3
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])

#4
loss = model.evaluate(x_test ,y_test)
print(loss)

x_pred = x_pred.reshape(1, 3, 1)

y_pred = model.predict(x_pred)
print(y_pred)

# [59.11595153808594, 6.207051753997803]
# [[81.39249]]

# [24.051712036132812, 3.5205540657043457]
# [[95.15763]]
