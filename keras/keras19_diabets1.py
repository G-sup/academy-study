# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape)
print(y.shape)

print(np.max(x), np.min(y))
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)

#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(120, input_dim=10, activation='relu'))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_test, y_test, epochs=1000, batch_size=6, validation_split=0.2, verbose=1)

#4
loss, mae = model.evaluate(x_test, y_test, batch_size=6)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# 기본
# loss,mae :  852.134521484375 14.923951148986816
# RMSE :  29.191344245671544
# R2:  0.852719952527542