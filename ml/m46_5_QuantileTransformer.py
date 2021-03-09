# 분위수를 1000개를 준다
# RobustScaler (중간분위수가 기준) 보다 이상치 제어의 효과가 있을 확률이 높다
# 아웃라이어 제거를 안해도 될 확률이 높다

import numpy as np

from sklearn.datasets import load_boston

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

scaler = QuantileTransformer() # 디폴트 : 균등분포
# scaler = QuantileTransformer(output_distribution='normal') # 정규분포

scaler.fit(x)
x = scaler.transform(x)

# QuantileTransformer
print(np.max(x), np.min(x))    # 1.0 0.0
print(np.max(x[0]))      # 1.0


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
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_split=0.2, verbose=1)

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


# RobustScaler
# loss,mae :  20.116552352905273 3.136497974395752
# RMSE :  4.48514798894428
# R2:  0.7616366647248813

# QuantileTransformer
# loss,mae :  17.520824432373047 2.5429718494415283
# RMSE :  4.185788359306913
# R2:  0.7923937466908131