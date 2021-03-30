import os
import os, glob
from PIL import Image
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization, Embedding, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.utils import to_categorical
from numpy import asarray

caltech_dir =  'C:/data/lotte/LPD_competition/train'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)

image_w = 256
image_h = 256

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("C:/data/lotte/npy/x_3D.npy", arr=X)
np.save("C:/data/lotte/npy/y_3D.npy", arr=y)

img1=[]
for i in range(0,72000):
    filepath='C:/data/lotte/LPD_competition/test/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((256,256))
    image_data2=asarray(image2)
    img1.append(image_data2)    

np.save('C:/data/lotte/npy/pred_x_3D.npy', arr=img1)

x = np.load("C:/data/lotte/npy/x_3D.npy",allow_pickle=True)
y = np.load("C:/data/lotte/npy/y_3D.npy",allow_pickle=True)

x_pred = np.load('C:/data/lotte/npy/pred_x_3D.npy',allow_pickle=True)


sub = pd.read_csv('C:/data/lotte/LPD_competition/sample.csv')


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=104)

print(x.shape)
print(y.shape)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_pred.shape)


model = Sequential()
model.add(Conv2D(32,2,1,input_shape = (256,256,3),padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,2,2,padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,2,2,padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128,2,1,padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,2,2,padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(512,2,2,padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(1560))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1280))
model.add(BatchNormalization())
model.add(Activation('relu'))
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