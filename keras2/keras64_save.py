# model.save
# pickle


import numpy as np
from numpy.core.fromnumeric import shape 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout

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
    outputs = Dense(10,activation='softmax',name='ousts')(x)
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

# 그냥 모델을 서치에 넣으면 안된다 랩핑안에 넣어서 돌려야 한다
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose = 1)
#  여기까지가 랩핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)



search.fit(x_train,y_train,verbose=1)

print(search.best_params_) # 내가 선택한 세개의 파라미터 {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}
print(search.best_estimator_) # 모든 파라미터에 대한 것(내가 튠하지 않은것도 나온다) 랩핑한거라 뒤 처럼 나온다 <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000023E8D35B670>
print(search.best_score_) # 0.9561999837557474 스코어랑 수치가 다르다

search.best_estimator_.model.save('../data/h5/keras64_pickle.h5') # 그냥 modelsave는 안먹힌다

acc = search.score(x_test,y_test) # acc :  0.9581999778747559
print('acc : ', acc)
import pickle

# pickle.dump(search, open('../data/xgb_save/keras64.pickle.dat','wb'))  # dump : save랑 같다
