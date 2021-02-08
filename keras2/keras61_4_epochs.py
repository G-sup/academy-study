# epoch 100
# validation_split, callback
# early_stopping 5
# Reduce LR 3
#  modelcheckpoint

import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

(x_train, y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test  = x_test.reshape(10000,28*28).astype('float32')/255.

#2 

# 함수형으로 만든 모델

def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28,), name='inputs')
    x = Dense(512,activation = activation,name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation = activation,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation = activation,name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name='ouputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')
    return model

# 함수형으로 만든 하이퍼 파라미터

def  create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4]
    activations = ['relu','tanh','sigmoid']
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop': dropout, 'activation' : activations}

hyperparameters = create_hyperparameters()
model2 = build_model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# 여기서 epochs, validation, callback등 넣을수 있다
model2 = KerasClassifier(build_fn=build_model,verbose = 1)#,epochs= 3,validation_split = 0.2)


search = RandomizedSearchCV(model2,hyperparameters,cv=3)

lr = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,verbose=1) 
modelpath = '../Data/modelCheckPoint/k61_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' 
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')

# epochs, validation, callback등 fit에서도 먹힌다 (fit이 우선순위가 더 높다)
search.fit(x_train,y_train,verbose=1,epochs=100 ,validation_split= 0.2, callbacks=[mc,es,lr])

print(search.best_params_) 
print(search.best_estimator_)
print(search.best_score_) 

acc = search.score(x_test,y_test) 
print('acc : ', acc)

# {'optimizer': 'adam', 'drop': 0.4, 'batch_size': 20, 'activation': 'sigmoid'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001D2DB370C40>
# 0.9788666566212972
# 500/500 [==============================] - 0s 892us/step - loss: 0.0707 - acc: 0.9841
# acc :  0.9840999841
