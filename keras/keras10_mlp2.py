# 다 : 1 mlp

import numpy as np

#1 데이터
x = np.array([range(100),range(301,401),range(1,101)])
y = np.array(range(711, 811))

print(x.shape) # (3, 100)
print(y.shape) # (100,)

x = np.transpose(x)
print(x)
print(x.shape) #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
print(x_train.shape)
print(y_train.shape)

#2 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) 

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2)

#4 평가 예측
loss,mae = model.evaluate(x_test, y_test)
print('loss : ',loss)
print('mae : ', mae)

y_predict = model.predict(x_test) #x가 아닌이유는 y_test와 shape가 틀리다
# print(y_predict)
# print(x_test.shape)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)


# loss :  0.01770561747252941
# mae :  0.12401855736970901

# loss :  2.495944562141972e-09
# mae :  3.356933666509576e-05

# RMSE :  0.023687287276830157  
# R2:  0.9999992602156662

# loss: 2.2501e-08 - mae: 1.3306e-04
# RMSE :  0.00014378929430407405
# R2:  0.9999999999708604

# loss :  2.9802322831784522e-09
# mae :  3.0517578125e-05
# RMSE :  5.9489711577203145e-05
# R2:  0.999999999995522