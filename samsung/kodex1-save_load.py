import numpy as np
from numpy.core.fromnumeric import shape, size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1

df = pd.read_csv('../study/samsung/KODEX.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 

a = df[['종가','고가','저가','금액(백만)','거래량','시가']]
a.columns = ['close','high','low','price(million)', 'volume','start'] # 열(columns)의 이름변경

a = a.loc[::-1]

print(a.shape)
x = a.iloc[424 : 1089,:]

# print(s1)
# print(s2)
# print(s3)
# print(s4)

print(x)

x = x.dropna(axis=0).values

def split_x(seq,size,col):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size),0:col].astype('float32')
        aaa.append(np.array(subset))
    return np.array(aaa)

size = 6
col = 6

dataset = split_x(x,size, col)


x = dataset[:-2,:7,:]
x_pred = dataset[-2:,:,:]

print(x.shape)
print(x_pred.shape)

x_train, x_test = train_test_split(x, train_size = 0.8, random_state=104)
x_train, x_val= train_test_split(x_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
x_pred = x_pred.reshape(1, -1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 6, 6)
x_train = x_train.reshape(-1, 6, 6)
x_test = x_test.reshape(-1, 6, 6)
x_val = x_val.reshape(-1, 6 ,6)
x_pred = x_pred.reshape(-1, 6 ,6)


print(x_train.shape) 
print(x_test.shape)

print(x.shape)
print(x_pred.shape)


# x = np.load('./samsung/s2.npy',allow_pickle=True)[0]
# y = np.load('./samsung/s2.npy',allow_pickle=True)[1]
# x_pred = np.load('./samsung/s2.npy',allow_pickle=True)[2]
# x_train = np.load('./samsung/s2.npy',allow_pickle=True)[3]
# x_test = np.load('./samsung/s2.npy',allow_pickle=True)[4]
# x_val = np.load('./samsung/s2.npy',allow_pickle=True)[5]
# y_train = np.load('./samsung/s2.npy',allow_pickle=True)[6]
# y_test = np.load('./samsung/s2.npy',allow_pickle=True)[7]
# y_val = np.load('./samsung/s2.npy',allow_pickle=True)[8]

np.save('./samsung/e-k1.npy',arr=([x,x_pred,x_train,x_test,x_val]))
