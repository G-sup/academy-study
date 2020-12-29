from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

'''
x_train = x[:60] # 순서가 0부터 59번째 까지 니까 1~60
x_val = x[60:80] # 61 ~ 80
x_test = x[80:]  # 81 ~ 100
#리스트의 슬라이싱

y_train = y[:60] # 순서가 0부터 59번째 까지 니까 1~60
y_val = y[60:80] # 61 ~ 80
y_test = y[80:]  # 81 ~ 100
'''

from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6 )
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

#2 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(1))


#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam',metrics='mae')
model.fit(x_train, y_train, epochs=100)

#4 평가 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss',loss)
print('mae', mae)

y_predict = model.predict(x_test)
print(y_predict)

#shuffle=False
#loss 0.0011465930147096515
#mae 0.03332405164837837

#shuffle=True
#loss 0.0027259509079158306
#mae 0.043111301958560944

