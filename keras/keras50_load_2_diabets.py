# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

#1 데이터

x = np.load('../data/npy/diabetes_x.npy')
y = np.load('../data/npy/diabetes_y.npy')

print(x[:5])
print(y[:10])
print(x.shape)
print(y.shape)

# print(np.max(x), np.min(y))
# print(dataset.feature_names)
# print(dataset.DESCR)


# x = x/442

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state= 104, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.3, random_state= 104, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(356,activation='relu', input_dim=10))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))


#3
from tensorflow.keras.callbacks import ModelCheckpoint ,EarlyStopping# callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k46_MC_5_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='val_loss',patience=30,mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping,mc])

#4
loss, mae = model.evaluate(x_test, y_test, batch_size=6)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# x_val
# loss,mae :  2761.839111328125 43.19102096557617
# RMSE :  52.55319959432148
# R2:  0.5226531608553822
