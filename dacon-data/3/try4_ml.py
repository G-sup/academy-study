import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential,Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten,Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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
# pred = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')

# print(y_train)
x_train,x_test,y_train,y_test = train_test_split (x_train,y_train,train_size=0.8, random_state=104)




def build_model(drop=0.5, optimizer='adam', activation='relu',activation1='relu'):
    inputs = Input(shape=(120,120,1), name='inputs')
    x = Conv2D(32, (2,2),padding='same', strides=1, input_shape=(120,120,1),activation=activation)(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2,2),padding='same', strides=2,activation=activation)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (2,2),padding='same', strides=2,activation=activation)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (2,2),padding='same', strides=2,activation=activation1)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64,activation=activation1)(x)
    x = BatchNormalization()(x)
    outputs = Dense(26,activation='sigmoid',name='ouputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer='adam',metrics=['acc'],loss='binary_crossentropy')
    return model

# 함수형으로 만든 하이퍼 파라미터

def  create_hyperparameters():
    batches = [8, 16, 32, 64, 128]
    dropout = [0.1, 0.2, 0.3, 0.4]
    activations = ['relu','tanh','sigmoid']
    return {'batch_size' : batches, 'drop': dropout, 'activation' : activations, 'activation1' : activations}

hyperparameters = create_hyperparameters()
model2 = build_model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose = 1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(model2,hyperparameters,cv=3)

lr = ReduceLROnPlateau(monitor='acc',patience=35, factor=0.5,verbose=1) 
es = EarlyStopping(monitor='acc',patience=80,mode='auto')
model.fit(x_train,y_train,verbose=1, epochs=1000, callbacks=[es,lr])

acc = model.score(x_test,y_test) 
print('acc : ', acc)

sub = model.predict(x_pred)

sub  = np.where(sub >0.5, 1, sub )
sub  = np.where(sub <0.5, 0, sub )

sample_submission = pd.read_csv('C:/data/dacon/dacon3/sample_submission.csv')
sample_submission.iloc[:,1:] = sub
sample_submission.to_csv("C:/data/dacon/dacon3/Dacon_3_ml.csv", index = False)
sample_submission