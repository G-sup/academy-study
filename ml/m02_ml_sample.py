
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# 1 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# OneHotEncoding 

# # sklearn 버젼
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# ohe.fit(y)
# y = ohe.transform(y).toarray()

# x, y = load_iris(return_X_y=True) 이런식으로 사용가능 위 세줄과 동일하다
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=104)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


print(y.shape) # (150, 3)
print(y)


#2 모델 구성
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(356, activation='relu', input_shape=(4,)))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(3,activation='softmax'))

model = LinearSVC()

#3 훈현

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='loss',patience=5,mode='auto')
# model.fit(x_train, y_train, epochs=250, batch_size=2, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping])
model.fit(x_train, y_train)

#4 평가 예측

# results = model.evaluate(x_test, y_test, batch_size=2)
results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x[-5:-1])

# 결과치 나오게 #argmax
# y_pred = np.argmax(y_pred,axis=-1)
# y_test = np.argmax(y_test,axis=-1)
print(y_pred)
print(y[-5:-1])