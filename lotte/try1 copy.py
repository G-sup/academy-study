import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization, Embedding, Activation,ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# train_datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.3,
#     validation_split = 0.2,
#     # preprocessing_function= preprocess_input,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(
#     # preprocessing_function= preprocess_input,
# )

# xy_train = train_datagen.flow_from_directory(
#     'C:/data/lotte/LPD_competition/train',
#     target_size=(128,128),
#     batch_size=50000,
#     seed=42,
#     subset='training',
#     class_mode='categorical'
# )  
# xy_test = train_datagen.flow_from_directory(
#     'C:/data/lotte/LPD_competition/train',
#     target_size=(128,128),
#     batch_size=50000,
#     seed=42,
#     subset='validation',
#     class_mode='categorical'
# )  
# x_pred = test_datagen.flow_from_directory(
#     'C:/data/lotte/LPD_competition/test',
#     target_size=(128,128),
#     batch_size=72000,
#     seed=42,
#     class_mode=None
# )   

# np.save('C:/data/lotte/npy/train_x.npy', arr=xy_train[0][0])
# np.save('C:/data/lotte/npy/train_y.npy', arr=xy_train[0][1])
# np.save('C:/data/lotte/npy/test_x.npy', arr=xy_test[0][0])
# np.save('C:/data/lotte/npy/test_y.npy', arr=xy_test[0][1])
# np.save('C:/data/lotte/npy/pred_x.npy', arr=x_pred[0])

sub = pd.read_csv('C:/data/lotte/LPD_competition/sample.csv')

x_train = np.load('C:/data/lotte/npy/train_x.npy')
y_train = np.load('C:/data/lotte/npy/train_y.npy')
x_test = np.load('C:/data/lotte/npy/test_x.npy')
y_test = np.load('C:/data/lotte/npy/test_y.npy')
x_pred = np.load('C:/data/lotte/npy/pred_x.npy')

x_train = x_train.reshape(x_train.shape[0], 128, 128, 3).astype('float32')
x_train = (x_train - 127.5) / 127.5 

x_test = x_test.reshape(x_test.shape[0], 128, 128, 3).astype('float32')
x_test = (x_test - 127.5) / 127.5 

x_pred = x_pred.reshape(x_pred.shape[0], 128, 128, 3).astype('float32')
x_pred = (x_pred - 127.5) / 127.5 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_pred.shape)

model = Sequential()
model.add(Conv2D(128,2,2,input_shape = (128,128,3),kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,2,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(16,2,2,kernel_initializer='he_normal'))
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

sub.to_csv('C:/data/lotte/LPD_competition/ss2.csv',index=False)