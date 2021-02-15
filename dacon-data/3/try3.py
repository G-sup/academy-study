import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D

# train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
#     rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
# pred_datagen = ImageDataGenerator(rescale=1./255)

# # image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# x_train = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/train',seed=104,target_size=(120, 120),batch_size=50000,color_mode='grayscale')#,subset="training")
# x_pred = pred_datagen.flow_from_directory('C:/data/dacon/dacon3/predict',seed=104,target_size=(120, 120),batch_size=5000,color_mode='grayscale')

# np.save('C:/data/image/brain/npy/dacon_train_x.npy', arr=x_train[0][0])
# np.save('C:/data/image/brain/npy/dacon_pred_x.npy', arr=x_pred[0][0])

x_train = np.load('C:/data/image/brain/npy/dacon_train_x.npy')
x_pred = np.load('C:/data/image/brain/npy/dacon_pred_x.npy')

y_train = pd.read_csv('C:/data/dacon/dacon3/dirty_mnist_2nd_answer.csv', index_col=0, header=0)
sub = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')
    
model = Sequential()


# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# show augmented image data
sample_data = x_train[100].copy()
sample = expand_dims(sample_data,0)
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    sample_batch = sample_generator.next()
    sample_image=sample_batch[0]
    plt.imshow(sample_image.reshape(120,120))

# cross validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=30,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=100, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(x_train, y_train) :
    
    mc = ModelCheckpoint('best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = x_train[train_index]
    x_valid = x_train[valid_index]    
    y_train = y_train[train_index]
    y_valid = y_train[valid_index]

    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(x_pred,shuffle=False)
    
    model = Sequential()

    model.add(Conv2D(64, (2,2),padding='same', strides=1, input_shape=(120,120,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2,2),padding='same', strides=1,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='tanh'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(2,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=1000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('best_cvision.h5')
    result += model.predict_generator(test_generator,verbose=True)/15
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


model.summary()

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