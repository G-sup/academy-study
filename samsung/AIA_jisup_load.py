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

x = np.load('./samsung/s1.npy',allow_pickle=True)[0]
y = np.load('./samsung/s1.npy',allow_pickle=True)[1]
x_pred = np.load('./samsung/s1.npy',allow_pickle=True)[2]
x_train = np.load('./samsung/s1.npy',allow_pickle=True)[3]
x_test = np.load('./samsung/s1.npy',allow_pickle=True)[4]
x_val = np.load('./samsung/s1.npy',allow_pickle=True)[5]
y_train = np.load('./samsung/s1.npy',allow_pickle=True)[6]
y_test = np.load('./samsung/s1.npy',allow_pickle=True)[7]
y_val = np.load('./samsung/s1.npy',allow_pickle=True)[8]

model = load_model('../data/modelCheckPoint/samsung_test_1_199-2286324.5000000.hdf5')

result = model.evaluate(x_test,y_test)
print('로드 체크 포인트_loss : ',result[0])
print('로드 체크 포인트_mse : ',result[1])

y_pred = model.predict(x_pred)
print(y_pred)
'''

# samsung_test_63-155337.0625000.hdf5
# 로드 체크 포인트_loss :  465243.875
# 로드 체크 포인트_mse :  316.6705017089844
# [[89889.766]]
로드 체크 포인트_loss :  292542.59375
로드 체크 포인트_mse :  288.64373779296875
[[89835.25

samsung_test_103-306237.0625000.hdf5
로드 체크 포인트_loss :  162924.453125
로드 체크 포인트_mse :  290.15924072265625
[[89771.664]

로드 체크 포인트_loss :  214736.015625
로드 체크 포인트_mse :  260.56243896484375
[[89556.62]]

'''