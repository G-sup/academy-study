import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x = np.load('./samsung/x.npy')

y = x[:,5]
x_pred = x[2396,:]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x_pred = x_pred.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 6, 1)
x_train = x_train.reshape(-1, 6, 1)
x_test = x_test.reshape(-1, 6, 1)
x_val = x_val.reshape(-1, 6 ,1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

# x = np.load('./samsung/s2.npy',allow_pickle=True)[0]
# y = np.load('./samsung/s2.npy',allow_pickle=True)[1]
# x_pred = np.load('./samsung/s2.npy',allow_pickle=True)[2]
# x_train = np.load('./samsung/s2.npy',allow_pickle=True)[3]
# x_test = np.load('./samsung/s2.npy',allow_pickle=True)[4]
# x_val = np.load('./samsung/s2.npy',allow_pickle=True)[5]
# y_train = np.load('./samsung/s2.npy',allow_pickle=True)[6]
# y_test = np.load('./samsung/s2.npy',allow_pickle=True)[7]
# y_val = np.load('./samsung/s2.npy',allow_pickle=True)[8]

model = load_model('../data/modelCheckPoint/samsung_test_103-306237.0625000.hdf5')

result = model.evaluate(x_test,y_test)
print('로드 체크 포인트_loss : ',result[0])
print('로드 체크 포인트_mse : ',result[1])

y_pred = model.predict(x_pred)
print(y_pred)
'''
# samsung_test_48-279427.4688.hdf5
# 로드 체크 포인트_loss :  400916.0625
# 로드 체크 포인트_acc :  300.9427185058594
# [[89404.266]]
로드 체크 포인트_loss :  337584.375
로드 체크 포인트_mse :  357.6504821777344
[[89666.99]]

# samsung_test_63-155337.0625000.hdf5
# 로드 체크 포인트_loss :  465243.875
# 로드 체크 포인트_mse :  316.6705017089844
# [[89889.766]]
로드 체크 포인트_loss :  292542.59375
로드 체크 포인트_mse :  288.64373779296875
[[89835.25

# samsung_test_48-172564.3593750.hdf5
# 로드 체크 포인트_loss :  409409.4375
# 로드 체크 포인트_mse :  317.59796142578125
# [[90107.61]]

로드 체크 포인트_loss :  262910.4375
로드 체크 포인트_mse :  274.4584045410156
[[90241.25]]

samsung_test_37-412920.4687500.hdf5
로드 체크 포인트_loss :  545706.0625
로드 체크 포인트_mse :  693.641845703125
[[89934.195]]

samsung_test_103-306237.0625000.hdf5
로드 체크 포인트_loss :  162924.453125
로드 체크 포인트_mse :  290.15924072265625
[[89771.664]

로드 체크 포인트_loss :  214736.015625
로드 체크 포인트_mse :  260.56243896484375
[[89556.62]]
'''