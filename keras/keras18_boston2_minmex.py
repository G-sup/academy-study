# 데이터 전처리 (MIN MAX)

import numpy as np

from sklearn.datasets import load_boston

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)
print('===================')
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))     # 711.0, 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MIN MAX)
# x = x /711.
# x = (x - min) / (max - min)
#   = (x - np.min(x)) / (np.max(x) - np.min(x))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104, shuffle=True)

#2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense

input1 = Input(shape=(13,))
dense1 = Dense(120, activation='relu')(input1)
dense1 = Dense(80)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(7)(dense1)
dense1 = Dense(5)(dense1)
dense1 = Dense(4)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=150, batch_size=1, validation_split=0.2, verbose=1)

#4평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# 전처리 전
# loss,mae :  10.771950721740723 2.4755585193634033
# RMSE :  3.2820651926340183
# R2:  0.8723619073719358

# 전처리 후
# loss,mae :  17.4603271484375 3.0702922344207764
# RMSE :  4.1785558930421125
# R2:  0.7931105569532743

# 전처리 전
# loss,mae :  13.33849048614502 2.7177727222442627
# RMSE :  3.6521900386363155
# R2:  0.8419506790710227

#전처리 후
# loss,mae :  13.422076225280762 2.818122625350952
# RMSE :  3.6636154345138077
# R2:  0.8409602592916448