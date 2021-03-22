import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization, Embedding, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
    zoom_range=1.2,shear_range=0.7,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
image_generator = ImageDataGenerator(rescale=1/255)   

xy_train = train_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',seed=104,target_size=(256, 256),batch_size=100000, subset="training",color_mode="grayscale")
xy_test = test_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',seed=104,target_size=(256, 256),batch_size=100000 , subset="validation",color_mode="grayscale")
x_pred = image_generator.flow_from_directory('C:/data/lotte/LPD_competition/test',target_size=(256, 256),batch_size=72000,color_mode="grayscale")

np.save('C:/data/lotte/npy/train_x.npy', arr=xy_train[0][0])
np.save('C:/data/lotte/npy/train_y.npy', arr=xy_train[0][1])
np.save('C:/data/lotte/npy/test_x.npy', arr=xy_test[0][0])
np.save('C:/data/lotte/npy/test_y.npy', arr=xy_test[0][1])
np.save('C:/data/lotte/npy/pred_x.npy', arr=x_pred[0][0])

sub = pd.read_csv('C:/data/lotte/LPD_competition/sample.csv')

x_train = np.load('C:/data/lotte/npy/train_x.npy')
y_train = np.load('C:/data/lotte/npy/train_y.npy')
x_test = np.load('C:/data/lotte/npy/test_x.npy')
y_test = np.load('C:/data/lotte/npy/test_y.npy')
x_pred = np.load('C:/data/lotte/npy/pred_x.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_pred.shape)

model = Sequential()
model.add(Conv2D(16,2,2,input_shape = (256,256,1),kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1000,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
es = EarlyStopping(monitor='val_loss',patience=50,verbose=1,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',patience=15,factor=0.05,verbose=1,mode='auto')
model.fit(x_train,y_train,epochs=10000,batch_size = 32,validation_split = 0.2, verbose=1,callbacks=[es,lr])

loss = model.evaluate(x_test,y_test)
print(loss)
y_pred = model.predict(x_pred)
print(y_pred.argmax(1))

sub['prediction'] = y_pred.argmax(1)

sub

sub.to_csv('C:/data/lotte/LPD_competition/ss2.csv',index=False)