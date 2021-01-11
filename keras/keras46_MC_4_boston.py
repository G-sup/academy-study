# x_val을 넣어보면 성능이 더 좋아진다( 안좋아 지는 경우도 있지만 데이터가 많을수록 더 좋아진다 )


import numpy as np

from sklearn.datasets import load_boston

#1 데이터 
dataset = load_boston()
x = dataset.data 
y = dataset.target
# print(x.shape) # (506, 13)
# print(y.shape) # (506,)
# print('===================')
# print(x[:5])
# print(y[:10])

# print(np.max(x), np.min(x))     # 711.0, 0.0
# print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MIN MAX)
# x = x /711.
# x = (x - min) / (max - min)
#   = (x - np.min(x)) / (np.max(x) - np.min(x))

# 민 맥스 스케일러 x만
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=104, shuffle=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, train_size=0.8, random_state=104, shuffle=True)

# 민 맥스 스케일러 x_train

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(np.max(x), np.min(x))    # 1.0, 0.0
print(np.max(x[0]))

#2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,Dropout

input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(64,activation='relu')(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(8)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping # callbacks 안에 넣어준다
modelpath = './modelCheckPoint/k46_MC_4_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_val, y_val), verbose=1,callbacks=[early_stopping,mc])

#4평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
print('loss,mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict ))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2: ', r2)

# 전처리후 MinMaxScailer x_train
# loss,mae :  9.900341987609863 2.169292688369751
# RMSE :  3.146480888810911
# R2:  0.8826897134778194

