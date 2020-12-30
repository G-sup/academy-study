import numpy as np

#1 데이터
x1 = np.array([range(100),range(301,401),range(1,101)])
y1 = np.array([range(711, 811), range(1,101), range(201,301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split 
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle=False, train_size=0.8)

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middlel1 = Dense(30)(merge1)
middlel1 = Dense(10)(middlel1)
middlel1 = Dense(10)(middlel1)

# 모델 분기 1
output1 = Dense(30)(middlel1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 모델 분기 2
output2 = Dense(15)(middlel1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])
model.summary()
'''
#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)

#4 평가 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('model.metrics_names : ', model.metrics_names)
print(loss)
 
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('======================================')
print('y1_predict : \n', y1_predict)
print('======================================')
print('y2_predict : \n', y2_predict)
print('======================================')

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE = (RMSE1 + RMSE2) /2
print('RMSE1 : ', RMSE1)
print('RMSE2 : ', RMSE2)
print('RMSE : ', RMSE)

#print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2 = (r2_1 + r2_2)/2

print('R2_1 : ', r2)
print('R2_2 : ', r2)
print('R2: ', r2)
'''
# y_predict = concatenate([y1_predict, y2_predict])
# y_test = concatenate([y1_test, y2_test])


# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE : ', RMSE(y_test, y_predict ))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('R2: ', r2)