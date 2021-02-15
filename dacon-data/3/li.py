import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten,MaxPool2D,Convolution2D, LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
#     rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
# pred_datagen = ImageDataGenerator(rescale=1./255)

# # image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# x_train = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/train',seed=104,target_size=(255, 255),batch_size=50000,color_mode='grayscale')#,subset="training")
# x_pred = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/predict',seed=104,target_size=(255, 255),batch_size=5000,color_mode='grayscale')

# np.save('C:/data/image/brain/npy/dacon_train_x_255.npy', arr=x_train[0][0])
# np.save('C:/data/image/brain/npy/dacon_pred_x_255.npy', arr=x_pred[0][0])

x_train = np.load('C:/data/image/brain/npy/dacon_train_x_255.npy')
x_pred = np.load('C:/data/image/brain/npy/dacon_pred_x_255.npy')

y_train = pd.read_csv('C:/data/dacon/dacon3/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
pred = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')

# print(y_train)
x_train,x_test,y_train,y_test = train_test_split (x_train,y_train,train_size=0.8, random_state=104)



model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(255,255,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4, 4)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4, 4)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4, 4)))


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(26,activation = 'sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
lr = ReduceLROnPlateau(monitor='val_loss',patience=35, factor=0.5,verbose=1) 
es = EarlyStopping(monitor='val_loss',patience=80,mode='auto')
model.fit(x_train,y_train,verbose=1,batch_size=16,epochs=1 ,validation_split= 0.2, callbacks=[es,lr])

model.evaluate(x_test,y_test)
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