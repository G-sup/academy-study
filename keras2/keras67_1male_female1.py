#  imagegrnerator의 fit_generator 사용

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
    rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

# image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

train = train_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=100 ,class_mode='binary', subset="training")
test = test_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=86 ,class_mode='binary', subset="validation")


model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
lr = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.5,verbose=1) 
es = EarlyStopping(monitor='val_loss',patience=30,mode='auto')
hist = model.fit_generator(train, steps_per_epoch=13,epochs=1000,validation_data=test,validation_steps=4,callbacks=[es,lr])
# steps_per_epoch 전체 데이터를 배치사이즈로 나눈값을 넣어 줘야한다 160개의 데이터를 5로 나눴으니까 32 
# 31개면 데이터 1를 버린거 33개면 안돌아간다

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['acc']
val_loss = hist.history['val_acc']

# 시각화
import matplotlib.pyplot as plt




plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss' , 'train acc', 'val acc'])
plt.show()

print(acc)

# Epoch 100/100
# 13/13 [==============================] - 6s 470ms/step - loss: 0.5044 - acc: 0.7448 - val_loss: 0.4414 - val_acc: 0.8110

# Epoch 00140: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
# 13/13 [==============================] - 6s 469ms/step - loss: 0.4533 - acc: 0.7789 - val_loss: 0.4355 - val_acc: 0.7994