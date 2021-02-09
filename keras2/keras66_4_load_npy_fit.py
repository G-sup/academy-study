# npy로 불러오기

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
x_train = np.load('C:/data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('C:/data/image/brain/npy/keras66_train_y.npy')

x_test = np.load('C:/data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('C:/data/image/brain/npy/keras66_test_y.npy')




model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
lr = ReduceLROnPlateau(monitor='val_loss',patience=25,factor=0.5,verbose=1) 
es = EarlyStopping(monitor='val_loss',patience=50,mode='auto')
model.fit(x_train,y_train,verbose=1,epochs=1000 ,validation_split= 0.2, callbacks=[es,lr])

acc = model.evaluate(x_test,y_test)
print(acc)