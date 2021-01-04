# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

#1 데이터
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


# x = x/442

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.3, random_state= 104, shuffle=True)

#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(120, input_dim=10))
model.add(Dense(120))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=6, validation_data=(x_val, y_val), verbose=1)

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
# loss,mae :  2696.805419921875 43.70464324951172
# RMSE :  51.93077936188863
# R2:  0.549811921815972

# x통짜
# loss,mae :  2693.2861328125 43.16354751586914
# RMSE :  51.8968816499351
# R2:  0.5503994487280122

# 민 맥스 스케일러 x 전체를 다
# loss,mae :  2814.070556640625 43.658714294433594
# RMSE :  53.047809880009815
# R2:  0.5302365472968072

# x_train 
# loss,mae :  2455.87841796875 41.00312042236328
# RMSE :  49.55682010723396
# R2:  0.5755343121533955

# x_val
# loss,mae :  36.10376739501953 5.0196919441223145
# RMSE :  6.008641561271572
# R2:  0.9937599463818423
