from keras_preprocessing.image import image_data_generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Conv1D ,Flatten, MaxPooling2D,MaxPooling1D,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.recurrent import GRU
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score,KFold,RandomizedSearchCV,StratifiedKFold
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD,Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

#1
train = pd.read_csv('./dacon-data/2/train.csv', index_col=[0,2], header=0) 
pred = pd.read_csv('./dacon-data/2/test.csv', index_col=[0,1], header=0) 

trian_datagen = ImageDataGenerator(rescale = 1/255,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.15,rotation_range = 10,vertical_flip=True)
test_datagen = ImageDataGenerator(rescale = 1/255)



y = train['digit'].values
x = train.drop(['digit'],1).values

x_pred = pred.values


# x_train, x_test, y_train,y_test = train_test_split(x,y, train_size = 0.8,random_state=104)
# x_train, x_val, y_train,y_val = train_test_split(x_train,y_train, train_size = 0.8,random_state=104)



x = x.reshape(-1, 28, 28,1)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# x_val = x_val.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)

# print(x_train.shape) 
# print(x_test.shape)
print(x_pred.shape)


# form keras.utils.np_utils import  to_categorical



def modeling() :

    model = Sequential()
    model.add(Conv2D(128, (2,2),padding='same', strides=1, input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2),padding='same', strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (2,2),padding='same', strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (2,2),padding='same', strides=1,activation='tanh'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128,activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(10,activation='softmax'))
    return model

#3

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0
t_d = train['digit'] # y 값 부여

optimizer =  Adam(lr=0.0002) 



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    print(x_train.shape, x_test.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
    print(y_train.shape, y_test.shape) #(1946,) (102,)

    print(y_train.shape, y_test.shape) #(1946,) (102,)

    train_generator = trian_datagen.flow(x_train, y_train, batch_size=32,seed=7,shuffle=True)
    valid_generator = test_datagen.flow(x_test,y_test,batch_size=32,seed=7,shuffle=True)
    pred_generator = test_datagen.flow(x_pred, batch_size=32,seed=7,shuffle=True)
    print(x_train.shape, x_test.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
    
    print(y_train.shape, y_test.shape) #(1946,) (102,)

    model = modeling()

    rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=30, factor=0.5,verbose=1)
    cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,filepath='./dacon-data/2/dacon2/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    es = EarlyStopping(monitor='val_loss',patience=100,mode='auto')

    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics='acc')
    hist = model.fit_generator(train_generator,steps_per_epoch=len(x_train)//32,epochs=1000,validation_data=valid_generator,validation_steps=4,verbose=1,callbacks = [es,rl,cp])
 
   
    result += model.predict_generator(pred_generator,verbose=True)/40 #a += b는 a= a+b

    print('result:', result)

    # save val_loss
    hist = pd.DataFrame(hist.history)
    val_loss_min.append(hist['val_loss'].min())
    nth += 1
    print(nth, 'set complete!!') # n_splits 다 돌았는지 확인

# print(val_loss_min, np.mean(val_loss_min))

model.summary()

#제출========================================
sub = pd.read_csv('./dacon-data/2/submission.csv')
sub['digit'] = result.argmax(1) # y값 index 2번째에 저장
sub
sub.to_csv('./dacon-data/2/0203.csv',index=False)

