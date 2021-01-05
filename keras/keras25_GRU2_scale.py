import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split

# 1
x = np.array([[1,2,3], [2,3,4,], [3,4,5], [5,6,7], [4,5,6], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

print(x.shape) #(13, 3)
print(y.shape) #(13,)

x = x.reshape(13, 3, 1)

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
model.add(GRU(356, input_shape=(3,1),activation='relu'))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=350, batch_size=1, validation_split=0.2, verbose=1)

#4
loss = model.evaluate(x_test ,y_test)
print(loss)

x_pred = x_pred.reshape(1, 3, 1)

y_pred = model.predict(x_pred)
print(y_pred)

# [118.51295471191406, 9.507526397705078]
# [[66.554375]]