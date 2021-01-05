# keras21_cancer1.py 를 다중분류로 코딩하시오

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# x, y = load_iris(return_X_y=True) 이런식으로 사용가능 위 세줄과 동일하다
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# OneHotEncoding 다중분류일때 사용 한다. 

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

print(x_train.shape) # (150, 4)
print(y_train.shape) # (150,)
# print(dataset.DESCR)
# print(dataset.feature_names)

# print(x[:5])
print(y)


#2 모델 구성
model = Sequential()
model.add(Dense(356, activation='relu', input_shape=(30,)))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(2,activation='softmax'))

#3 컴파일 훈현
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=250, batch_size=2, validation_split=0.2, verbose=1)

#4 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

y_pred = model.predict(x_test)
print(y_pred)

# 결과치 나오게 #argmax
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)
y_test = np.argmax(y_test,axis=1)
print(y_test)


# 하기전에도 될 것 같은 느낌이 났습니다. 해보니까 전혀 문제 없는 것 처럼 보이는데 이게 맞는건지는 잘모르겠습니다...