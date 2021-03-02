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

train = pd.read_csv('C:/data/dacon/dacon3/mnist_data/train.csv')
test = pd.read_csv('C:/data/dacon/dacon3/mnist_data/test.csv')
sub = pd.read_csv('C:/data/dacon/dacon3/mnist_data/submission.csv')

# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
test2 = test2/255.0

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# show augmented image data
sample_data = train2[100].copy()
sample = expand_dims(sample_data,0)
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    sample_batch = sample_generator.next()
    sample_image=sample_batch[0]
    plt.imshow(sample_image.reshape(28,28))

# cross validation
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=30,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=100, verbose=1)

val_loss_min = []
result = 0
nth = 0

y = train['letter']
y1=np.array(y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
print(y.shape)
y = encoder.fit(y1.reshape(-1,1)).toarray()

for train_index, valid_index in skf.split(train2,y) :
    
    mc = ModelCheckpoint('best_vision.h5',save_best_only=True, verbose=1)
    # y = encoder.fit_transform(y1.reshape(-1,1)).toarray()
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = y[train_index]
    y_valid = y[valid_index]

    # y_train = np.array(y_train)
    # y_train = encoder.transform(y_train.reshape(-1,1)).toarray()
    # y_valid = np.array(y_valid)
    # y_valid = encoder.transform(y_valid.reshape(-1,1)).toarray()   

    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = Sequential()

    model.add(Conv2D(64, (2,2),padding='same', strides=1, input_shape=(28,28,1),activation='relu'))
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
    model.add(Dense(26,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=1000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('best_vision.h5')
    result += model.predict_generator(test_generator,verbose=True)/15
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


model.summary()

sub['digit'] = result.argmax(1)

sub

sub.to_csv('C:/data/dacon/dacon3/mnist_data//Dacon_3.csv',index=False)