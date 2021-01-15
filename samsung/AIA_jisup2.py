import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dropout,Dense,GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1
df = pd.read_csv('../study/samsung/samsung.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 
df1 = pd.read_csv('../study/samsung/samsung2.csv',encoding='cp949',thousands = ',', index_col=0, header=0) 

a = df[['시가','고가','저가','금액(백만)','거래량','종가']]
a1 = df1[['시가','고가','저가','금액(백만)','거래량','종가']]

a.columns = ['start','high','low','price(million)', 'volume','close'] # 열(columns)의 이름변경
a1.columns = ['start','high','low','price(million)', 'volume','close'] # 열(columns)의 이름변경


print(a)
print(a.info())

a = a.loc[::-1]
a1 = a1.loc[::-1]

print(a1)
print(a1.tail()) # 디폴트 5 = df[-5:]
print(a1.info())
print(a1.describe())

print(a)
print(a.info())
x1 = a1.dropna(axis=0)

print(x1)

# a = pd.to_datetime(a)
# a = pd.to_numeric(a)

# s1 = a.iloc[0 : 1738,:].astype(float)
s1 = a.iloc[0 : 1738,0:4].astype('float')/50.
s2 = a.iloc[1738:,:]
s3 = a.iloc[0 : 1738,4]
s4 = a.iloc[0 : 1738,5].astype('float')/50.
s5 = pd.concat([s1,s3,s4],axis=1)
s6 = x1.iloc[[1],:].astype('float')

# print(s1)
# print(s2)
# print(s3)
# print(s4)

c = pd.concat([s5,s2,s6])

# x = np.load('./samsung/s2.npy',allow_pickle=True)[0]
x = c.dropna(axis=0).values


def split_x(seq, size, col) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 5
col = 6

dataset = split_x(x,size, col)

#1. DATA
x = dataset[:-1,:,:7]
y = dataset[1:,-1:,:1]

x_pred = dataset[-1:,:,:]



print(x.shape)
print(y.shape)
print(x_pred.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, random_state=104)

x = x.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_val = x_val.reshape(-1,1)
x_pred = x_pred.reshape(-1, 1)

# print(x_train.shape)    #(1533, 6)
# print(x_val.shape)      #(384, 6)
# print(x_test.shape)     #(480, 6)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

# print(x_train.shape) #(283, 13)
# print(x_test.shape) #(152,13)

x = x.reshape(-1, 6, 5)
x_train = x_train.reshape(-1, 6, 5)
x_test = x_test.reshape(-1, 6,5)
x_val = x_val.reshape(-1, 6 ,5)
x_pred = x_pred.reshape(1,6 ,5)



#2
model= Sequential()
model.add(GRU(512,activation='relu',input_shape=(6,5)))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#3
modelpath = '../data/modelCheckPoint/samsung_split_test_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='val_loss',patience=50,mode='auto')
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,epochs=1000,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping])#,mc])


#4
loss = model.evaluate(x_test,y_test,batch_size=16)
print(loss)

y_pred = model.predict(x_pred)
print(y_pred)
