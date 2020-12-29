import numpy as np

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])

y = np.array([1,2,3,4,5,6,7,8,9,10])
# 스칼라가 10개짜리 벡터비슷한거
print(x.shape) # (10,) = 스칼라가 10개  # (10,2) = 10행 2열(컬럼이 두개)

#x = x.T     #행렬변환
x = np.transpose(x)
print(x)
print(x.shape)

# ex) 행렬
# [[1,2,3],[4,5,6]]             = (2,3)
# [[1,2],[3,4],[5,6]]           = (3,2)
# [[[1,2,3],[4,5,6]]]           = (1,2,3)
# [[1,2,3,4,5,6]]               = (1,6)
# [[[1,2],[3,4]],[[5,6][7,8]]]  = (2,2,2)
# [[1],[2],[3]]                 = (3,1)
# [[[1],[2]],[[3],[4]]]         = (2,2,1)

# print(np.array([[1,2,3],[4,5,6]]).shape)
# print(np.array([[1,2],[3,4],[5,6]]).shape)
# print(np.array([[[1,2,3],[4,5,6]]]).shape)
# print(np.array([[1,2,3,4,5,6]]).shape)
# print(np.array([[[1,2],[4,5]],[[5,6],[7,8]]]).shape)
# print(np.array([[1],[2],[3]]).shape)
# print(np.array([[[1],[2]],[[3],[4]]]).shape)


#2 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense #조금느리다

model = Sequential()
model.add(Dense(10, input_dim=2)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) 

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1, validation_split=0.2)

#4 평가 예측
loss,mae = model.evaluate(x, y)
print('loss : ',loss)
print('mae : ', mae)

y_predict = model.predict(x)
# print(y_predict)

'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)
'''