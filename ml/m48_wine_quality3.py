import pandas as pd
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

wine = pd.read_csv('C:/data/Csv/winequality-white.csv', sep = ";", header=0)
# test = pd.read_csv('C:/data/Csv/data-01-test-score.csv',header=None)

x = wine.iloc[:,:-1].values
y = wine.iloc[:,11].values

# y = wine['quality'].values
# x = wine.drop('quality', axis=1).values


# 새로운 카테고리로 만들어 카테고리를 줄인다
# 7 > 3 개로 만든다 

newlist = []

for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]

y = newlist

# ======== 머신러닝에는 필요없음============
# y = np.array(y)

# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# ohe.fit(y)
# y = ohe.transform(y).toarray()

# print(y.shape)
# ========================================

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=104)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.7, random_state=104)

print(x_train.shape)
print(x_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)



# model = Sequential()

# model.add(Dense(1024,activation='relu',input_dim=11))
# model.add(Dropout(0.3))
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(3,activation='softmax'))

# # lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.1,verbose=1) 
# # es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')

# lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=20,mode='auto')
# es = EarlyStopping(monitor='val_loss',patience=45,mode='auto')
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train,y_train,epochs=1000,batch_size=4,validation_split=0.2,callbacks=[es,lr] )

# loss = model.evaluate(x_test,y_test)
# print(loss)


model = RandomForestClassifier()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

