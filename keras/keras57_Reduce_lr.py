
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
(x_train, y_train), (x_test,  y_test) = mnist.load_data()



x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# (x_test.reshpe(x_test.shape[0], x_test.shape[1]. x_test.shape[2], 1))


from tensorflow.keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)



# 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout

model = Sequential()
model.add(Conv2D(128, (2,2),padding='same', strides=2, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(1, (2,2),padding='same', strides=1))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(1, (2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(1, (2,2)))
model.add(Flatten())
model.add(Dense(1))
model.add(Dense(1000,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

#3

from tensorflow.keras.callbacks import ReduceLROnPlateau # callbacks 안에 넣어준다
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5,verbose=1) # factor = 0.5 : RL를 50%로 줄이겠다


modelpath = '../Data/modelCheckPoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' 
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
hist = model.fit(x_train, y_train, epochs=60,batch_size=64,validation_split=0.5,verbose=1,callbacks=[early_stopping,mc,reduce_lr])

#4
loss = model.evaluate(x_test,y_test)
print(loss[0])
print(loss[1])

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 면적을 찾아라 

plt.subplot(2,1,1) # (2,1) 짜리 1번째 그림

plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')

plt.grid() # 모눈종이형 격자 = grid


plt.title('mnist_loss n val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') 
# loc = 로케이션 , 라벨의 이름을 표기, 위치가 없을때는 빈공간에 자리한다 

# ================

plt.subplot(2,1,2) # (2,1) 짜리 2번째 그림 shbplt = 여러장을 보려할떄

plt.plot(hist.history['acc'],marker='.',c='red')
plt.plot(hist.history['val_acc'],marker='.',c='blue')

# plt.plot(hist.history['acc'],marker='.',c='red', label='acc')
# plt.plot(hist.history['val_acc'],marker='.',c='blue', label='val_acc')

plt.grid() 

plt.title('mnist_acc n val_acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) # 딕셔너리로 바로 사용가능

# plt.legend(loc='upper right') 

plt.show()

