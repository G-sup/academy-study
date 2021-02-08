# model.save
# pickle


import numpy as np
from numpy.core.fromnumeric import shape 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline

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
    return {'mo__batch_size' : batches, 'mo__optimizer' : optimizers, 'mo__drop': dropout, 'mo__activation' : activations}

hyperparameters = create_hyperparameters()
model2 = build_model

# 그냥 모델을 서치에 넣으면 안된다 랩핑안에 넣어서 돌려야 한다
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose = 1)
#  여기까지가 랩핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2,hyperparameters,cv=3)

pipe = Pipeline([('scaler',MinMaxScaler()),('mo',model2)]) 
# 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다  이름을 정해줘야 위에 Parameters를 조정가능하다(mo__:이름으로 지정) 


# model = GridSearchCV(pipe,Parameters,cv = 5)
model = RandomizedSearchCV(pipe,hyperparameters,cv = 5)

model.fit(x_train, y_train)

results = model.score(x_test,y_test)

print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 
print(model.best_params_) # 내가 선택한 세개의 파라미터 {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}
print(results)


# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mo',
#                  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000224CC201D60>)])
# {'mo__optimizer': 'rmsprop', 'mo__drop': 0.1, 'mo__batch_size': 40, 'mo__activation': 'tanh'}
# 0.9510999917984009