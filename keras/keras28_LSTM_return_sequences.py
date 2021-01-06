import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# 1
x = np.array([[1,2,3], [2,3,4,], [3,4,5], [5,6,7], [4,5,6], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)

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
model.add(LSTM(356, input_shape=(3,1),activation='relu',return_sequences=True))
# return_sequences=True = 3D로 입력을 시켜야 LSTM이 기동하기 떄문에 바꿔 줘야 한다 ex) = (none, 3, 356)
model.add(LSTM(356))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))
model.summary()

'''
#3
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=350, batch_size=1, validation_split=0.2, verbose=1)

#4
loss = model.evaluate(x_test ,y_test)
print(loss)

x_pred = x_pred.reshape(1, 3, 1)

y_pred = model.predict(x_pred)
print(y_pred)

# LSTM 1
# [48.46060562133789, 5.1912946701049805]
# [[85.34504]]

# LSTM 2
# [756.09033203125, 23.408042907714844]
# [[32.300705]]

# LSTM 3
# [1391.190185546875, 31.768936157226562]
# [[18.039225]]
'''