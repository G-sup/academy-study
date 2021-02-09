#  imagegrnerator의 fit 사용

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
    rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

# image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# xy_train = train_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=2000 ,class_mode='binary', subset="training")
# xy_test = test_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=2000 ,class_mode='binary', subset="validation")

# np.save('C:/data/npy/keras67_train_x.npy', arr=xy_train[0][0])
# np.save('C:/data/npy/keras67_train_y.npy', arr=xy_train[0][1])
# np.save('C:/data/npy/keras67_test_x.npy', arr=xy_test[0][0])
# np.save('C:/data/npy/keras67_test_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/npy/keras67_train_x.npy')
y_train = np.load('C:/data/npy/keras67_train_y.npy')

x_test = np.load('C:/data/npy/keras67_test_x.npy')
y_test = np.load('C:/data/npy/keras67_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=(100,100,3),activation='relu'))
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
model.fit(x_train,y_train,verbose=1,batch_size=4,epochs=1000 ,validation_split= 0.2, callbacks=[es,lr])

acc = model.evaluate(x_test,y_test)
print(acc)
