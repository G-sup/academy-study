# 과제 및 실습 
# 데이터 1 ~ 100 
# 5개씩
# 95,96,97,98,99    100
# predict를 만들것
# 96,97,98,99,100   >101   ~   100,101,102,103,104   > 105 
#  예상 프레딕트 101 102 103 104 105

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1
a = np.array(range(1,101))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) #[item for item in subset]
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
# print(dataset)
x = dataset[:, :5] 
y = dataset[:, 5] 


# x_pred = np.array([dataset[-1,1:],dataset[-1,1:],dataset[-1,1:],dataset[-1,1:]])
x_pred = np.array([dataset[-1,1:],[97,98,99,100,101],[98,99,100,101,102],[99,100,101,102,103],[100,101,102,103,104]])
# x_pred = dataset[-1,1:]

# print(x) #6,4 
# print(x.shape) #6,4 
# print(y) #6,
# print(y.shape) #4,

# x = x.reshape(-1, 5, 1)

# print(x_pred.shape)
# print(x.shape) #6,4 


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=104)


from sklearn.preprocessing import MinMaxScaler
x_pred = x_pred.reshape(-1, 5)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_val = scaler.transform(x_val)

print(x_train.shape) #(9, 3)
print(x_test.shape) #(4,3)

x = x.reshape(-1, 5, 1)
x_train = x_train.reshape(-1, 5, 1)
x_test = x_test.reshape(-1, 5, 1)
x_val = x_val.reshape(-1, 5 ,1)



#2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
 
model = Sequential()
model.add(LSTM(356, activation='relu' ,input_shape=(5,1)))
model.add(Dense(356))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3
model.compile(loss='mse',optimizer='adam',metrics='mse')
early_stopping = EarlyStopping(monitor='loss',patience=50,mode='auto')
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val,y_val), verbose=1,callbacks=[early_stopping])
#4
loss = model.evaluate(x_test,y_test,batch_size=2)

print(loss)

x_pred = x_pred.reshape(-1,5,1)
y_pred = model.predict(x_pred)

print(y_pred)


# [0.002878846600651741, 0.002878846600651741]
# [[100.92611 ]
#  [101.905846]
#  [102.88108 ]
#  [103.85714 ]
#  [104.833824]]


# Epoch 112/300
# 60/60 [==============================] - 0s 5ms/step - loss: 0.1422 - mse: 0.1422 - val_loss: 0.1568 - val_mse: 0.1568
# 10/10 [==============================] - 0s 1ms/step - loss: 0.1188 - mse: 0.1188
# [0.11878190189599991, 0.11878190189599991]
# [[100.963806]
#  [101.96486 ]
#  [102.96591 ]
#  [103.96701 ]
#  [104.96811 ]]