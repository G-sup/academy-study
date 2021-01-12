# 이중분류, 얼리스탑핑

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')

# print(x.shape) # (569, 30)
# print(y.shape) # (569,)
# print(x[:5])
# print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=104)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x.shape) 
print(y.shape)

#2 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(356,activation='relu', input_shape=(30,)))
model.add(Dropout(0.4))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dense(1, activation='sigmoid'))

#3 컴파일 훈련

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
# : mse=mean_squared_error, acc=accuracy 같이 풀네임 사용가능
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k46_MC_6_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping,mc])

#4 평가 예측

loss = model.evaluate(x_test, y_test, batch_size=2 )
print(loss)


# 실습 1 acc 0.985 이상
# 살습 2 predict 출력

# y[-5:-1] = ? 0 아니면 1

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])


# 결과치 나오게 코딩할것
 
# y_pred = np.argmax(y_pred,axis=1)
# print(y_pred)

# y_pred = model.predict_classes(x_test[-5:-1])
# y_pred = np.transpose(y_pred)
# print(y_pred)

y_pred = np.where(y_pred>0.5, 1, y_pred)
y_pred = np.where(y_pred<0.5, 0, y_pred)
y_pred = np.transpose(y_pred)
print(y_pred)

# Epoch 41/1000
# 182/182 [==============================] - 0s 685us/step - loss: 0.0763 - accuracy: 0.9698 - val_loss: 0.2062 - val_accuracy: 0.9560
# 57/57 [==============================] - 0s 6