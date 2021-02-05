import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Conv1D ,Flatten, MaxPooling2D,MaxPooling1D,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.recurrent import GRU
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score,KFold,RandomizedSearchCV
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD,Nadam

#1
train = pd.read_csv('./dacon-data/2/train.csv', index_col=[0,2], header=0) 
pred = pd.read_csv('./dacon-data/2/test.csv', index_col=[0,1], header=0) 
# sub = pd.read_csv('./dacon-data/2/submission.csv', index_col=0, header=0) 

y = train['digit'].values
x = train.drop(['digit'],1).values

x_pred = pred.values

pca =PCA(n_components=100) # n_components = : 컬럼을 몇개로 줄이는지

x_train, x_test, y_train,y_test = train_test_split(x,y, train_size = 0.8,random_state=104)
x_train, x_val, y_train,y_val = train_test_split(x_train,y_train, train_size = 0.8,random_state=104)

pca.fit(x_train)
x = pca.transform(x)
x_pred = pca.transform(x_pred)
x_test = pca.transform(x_test)
x_train = pca.transform(x_train)
x_val = pca.transform(x_val)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 10, 10)
x_train = x_train.reshape(-1, 10, 10)
x_test = x_test.reshape(-1, 10, 10)
# x_val = x_val.reshape(-1, 10, 10)
x_pred = x_pred.reshape(-1, 10, 10)

print(x_train.shape) 
print(x_test.shape)
print(x_pred.shape)

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)

model = Sequential()
model.add(Conv1D(128, (2,),padding='same', strides=1, input_shape=(10,10),activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(128, (2,),padding='same', strides=1,activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, (2,),padding='same', strides=1,activation='tanh'))
model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(BatchNormalization())
model.add(GRU(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))
model.summary()
optimizer =  Adam(lr=0.005)

#3
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics='acc')
rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=30, factor=0.5)
cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,filepath='./dacon-data/2/dacon2/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
es = EarlyStopping(monitor='val_loss',patience=100,mode='auto')
# model.fit(x_train, y_train, epochs=1,batch_size=164,validation_data=(x_val,y_val),verbose=1,callbacks = [es,rl,cp])
model.fit(x_train, y_train, epochs=1000,batch_size=8,validation_split = 0.2,verbose=1,callbacks = [es,rl,cp])

#4
loss = model.evaluate(x_test,y_test)
print(loss)
pred = model.predict(x_pred)
y_pred = np.argmax(pred,axis=-1)
print(y_pred[:10])

y_pred=pd.DataFrame(y_pred)
file_path='./dacon-data/2/result/result_ml_dl'+'.csv'
y_pred.to_csv(file_path)

submission = pd.read_csv('./dacon-data/2/submission.csv')
submission['digit'] = y_pred
submission.head()
submission.to_csv('./dacon-data/2/result/result_ml_dl'+'.csv', index=False)
