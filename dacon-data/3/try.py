import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
    rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator()
pred_datagen = ImageDataGenerator(rescale=1./255)

# image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# x_train = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/train',seed=104,target_size=(120, 120),batch_size=50000,color_mode='grayscale')#,subset="training")
# x_pred = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/predict',seed=104,target_size=(120, 120),batch_size=5000,color_mode='grayscale')

# np.save('C:/data/image/brain/npy/dacon_train_x.npy', arr=x_train[0][0])
# np.save('C:/data/image/brain/npy/dacon_pred_x.npy', arr=x_pred[0][0])

x_train = np.load('C:/data/image/brain/npy/dacon_train_x.npy')
x_pred = np.load('C:/data/image/brain/npy/dacon_pred_x.npy')

y_train = pd.read_csv('C:/data/dacon/dacon3/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
pred = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')



# print(y_train)
x_train,x_test,y_train,y_test = train_test_split (x_train,y_train,train_size=0.8, random_state=104)
x_train,x_val,y_train,y_val = train_test_split (x_train,y_train,train_size=0.8, random_state=104)

xy_train = train_datagen.flow(x_train,y_train,seed=104,batch_size=50)
xy_test = test_datagen.flow(x_train,y_train,seed=104,batch_size=500)
xy_val= test_datagen.flow(x_val,y_val,seed=104,batch_size=50)



model = Sequential()
model.add(Conv2D(64, (2,2),padding='same', strides=1, input_shape=(120,120,1),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2),padding='same', strides=1,activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3),padding='same', strides=2,activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# model.add(Conv2D(256, (3,3),padding='same', strides=2,activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(256, (3,3),padding='same', strides=2,activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(26,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
lr = ReduceLROnPlateau(monitor='val_loss',patience=35, factor=0.5,verbose=1) 
es = EarlyStopping(monitor='val_loss',patience=80,mode='auto')
model.fit_generator(xy_train, steps_per_epoch=100,epochs=1,validation_data=xy_val,validation_steps=4,callbacks=[es,lr])

print(model.evaluate(xy_test))

sub = model.predict(x_pred)

sub  = np.where(sub >0.5, 1, sub )
sub  = np.where(sub <0.5, 0, sub )

print(sub)
print(sub.shape)
# sub = pd.DataFrame(sub) 

# sub.to_csv('C:/data/dacon/dacon3/Dacon_3.csv',columns=pred.iloc[0],index=pred['index'],header='index')
sample_submission = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')
sample_submission.iloc[:,1:] = sub
sample_submission.to_csv("C:/data/dacon/dacon3/Dacon_3.csv", index = False)
sample_submission