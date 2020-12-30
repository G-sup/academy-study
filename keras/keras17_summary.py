import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='linear' ))
model.add(Dense(5, activation='linear', name='asdasd'))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(9))
model.add(Dense(1))

model.summary()

# 실습2 + 과제
# ensembel1, 2, 3, 4에대해 서머리를 계산하고 이해한것을 과제로 제출
# layer를 만들떄 'name' 이란놈에 대해 확인하고 설명 할것 , and 반드시 쓸 경우도 확인