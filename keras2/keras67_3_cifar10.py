import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

(x_train, y_train), (x_test,  y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
    zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator
xy_train = train_datagen.flow(x_train, y_train,batch_size=500)
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow(x_test,  y_test,batch_size=500)
# Found 120 images belonging to 2 classes.


model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
lr = ReduceLROnPlateau(monitor='val_loss',patience=20,factor=0.5,verbose=1) # factor = 0.5 : RL를 50%로 줄이겠다
es = EarlyStopping(monitor='val_loss',patience=40,mode='auto')
hist = model.fit_generator(xy_train, steps_per_epoch=100 ,epochs=1000,validation_data=xy_test,validation_steps=20,callbacks=[es,lr])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['acc']
val_loss = hist.history['val_acc']