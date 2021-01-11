# 다중 분류, 얼리스탑핑

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
# 1 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# OneHotEncoding 다중분류일때 사용 한다. 

# sklearn 버젼
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
ohe.fit(y)
y = ohe.transform(y).toarray()

# x, y = load_iris(return_X_y=True) 이런식으로 사용가능 위 세줄과 동일하다
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=104)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# tensorflow버전

# from tensorflow.keras.utils import to_categorical
# # form keras.utils.np_utils import  to_categorical
# y = to_categorical(y)
# y_test = to_categorical(y_test)
# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)

# print(x.shape) # (150, 4)
# print(y.shape) # (150, 3)
# print(dataset.DESCR)
# print(dataset.feature_names)

# print(x[:5])
# print(y)


# print(y.shape) # (150, 3)
# print(y)

print(y.shape) # (150, 3)
print(y)


#2 모델 구성
model = Sequential()
model.add(Dense(356, activation='relu', input_shape=(4,)))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3 컴파일 훈현
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = './modelCheckPoint/k46_MC_7_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss',patience=20,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping,mc])

#4 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print(loss)

y_pred = model.predict(x_test)

# 결과치 나오게 #argmax
y_pred = np.argmax(y_pred,axis=-1)
y_test = np.argmax(y_test,axis=-1)
print(y_pred)
print(y_test)

# Epoch 46/1000
# 48/48 [==============================] - 0s 1ms/step - loss: 0.1244 - acc: 0.9583 - val_loss: 0.0769 - val_acc: 1.0000
# 15/15 [==============================] - 0s 465us/step - loss: 0.0583 - acc: 0.9667
# [0.05825520306825638, 0.96666663885116