import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Conv1D ,Flatten, MaxPooling2D,MaxPool1D,Dropout,BatchNormalization,LeakyReLU
from tensorflow.keras.models import Sequential,load_model
from tensorflow.python.keras.backend import dropout
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD,Nadam

#1
train = pd.read_csv('./dacon-data/2/train.csv', index_col=[0,2], header=0) 
pred = pd.read_csv('./dacon-data/2/test.csv', index_col=[0,1], header=0) 
sub = pd.read_csv('./dacon-data/2/submission.csv', index_col=0, header=0) 


# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')

y = train['digit'].values
x = train.drop(['digit'],1).values


x_pred = pred.values
print(x_pred)
print(x.shape)
print(y.shape)

x_train, x_test, y_train,y_test = train_test_split(x,y, train_size = 0.8,random_state=104)
x_train, x_val, y_train,y_val = train_test_split(x_train,y_train, train_size = 0.8,random_state=104)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape) 
print(x_test.shape)

x = x.reshape(-1, 28, 28,1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)

print(x_train.shape) 
print(x_test.shape)
print(x_pred.shape)

from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print(x_train.shape, x_test.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
print(y_train.shape, y_test.shape) #(1946,) (102,)

model = Sequential()
model.add(Conv2D(128, (2,2),padding='same', strides=1, input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(256, (2,2),padding='same', strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(256, (2,2),padding='same', strides=1,activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128,activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))
model.summary()

#3
optimizer =  Adam(lr=0.0002) 

model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics='acc')
rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=40, factor=0.5)
cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,filepath='./dacon-data/2/dacon2/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
es = EarlyStopping(monitor='val_loss',patience=120,mode='auto')
# model.fit(x_train, y_train, epochs=1,batch_size=164,validation_data=(x_val,y_val),verbose=1,callbacks = [es,rl,cp])
hist = model.fit(x_train, y_train, epochs=1000,batch_size=4,validation_data=(x_val,y_val),verbose=1,callbacks = [es,rl,cp])

#4
loss = model.evaluate(x_test,y_test)
print(loss)
pred = model.predict(x_pred)
# y_pred = np.argmax(pred,axis=-1)
# print(y_pred[:10])


# y_pred=pd.DataFrame(y_pred)
# file_path='./dacon-data/2/result/result_'+'.csv'
# y_pred.to_csv(file_path)

# [2.3260953426361084, 0.44634145498275757]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss' , 'train acc', 'val acc'])
plt.show()

sub = pd.read_csv('./dacon-data/2/submission.csv')
sub['digit'] = pred.argmax(1) # y값 index 2번째에 저장
sub
sub.to_csv('./dacon-data/2/0203.csv',index=False)