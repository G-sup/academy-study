
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
# 1 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# sklearn 버젼
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
ohe.fit(y)
y = ohe.transform(y).toarray()

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
model = Sequential()
model.add(Dense(356, activation='relu', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(3,activation='softmax'))

#3 컴파일 훈현
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x_train, y_train, epochs=250, batch_size=2, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping])

#4 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

y_pred = model.predict(x_test)

# 결과치 나오게 #argmax
y_pred = np.argmax(y_pred,axis=-1)
y_test = np.argmax(y_test,axis=-1)
print(y_pred)
print(y_test)


# [0.08409680426120758, 0.9666666388511658]


# [0.07874599099159241, 0.9666666388511658]