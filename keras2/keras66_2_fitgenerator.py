import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential


train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
    zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory(폴더) 에서 변환

# train_generator
xy_train = train_datagen.flow_from_directory('C:/data/image/brain/train',target_size=(150, 150),batch_size=5 ,class_mode='binary')
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory('C:/data/image/brain/test',target_size=(150, 150),batch_size=5,class_mode='binary')
# Found 120 images belonging to 2 classes.


model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(150,150,3),activation='relu'))
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

hist = model.fit_generator(xy_train, steps_per_epoch=32,epochs=100,validation_data=xy_test,validation_steps=4)
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
