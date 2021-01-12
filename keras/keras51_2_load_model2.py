
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
(x_train, y_train), (x_test,  y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


from tensorflow.keras.utils import to_categorical
# form keras.utils.np_utils import  to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# 2
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout

'''
model = Sequential()
model.add(Conv2D(128, (2,2),padding='same', strides=2, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (2,2),padding='same', strides=1))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(128, (2,2)))
model.add(Conv2D(128, (2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()

model.save("./model/keras51_1_model1.h5")

#3

from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = './modelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')
# filepath 그지점에 W 값이 들어간다

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='val_loss',patience=5,mode='auto')
hist = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

model.save("./model/keras51_1_model2.h5")
'''


model = load_model('../data/h5/keras51_1_model2.h5')
# 컴파일 훈련도 같이 저장가능

#4
loss = model.evaluate(x_test,y_test)
print(loss[0])
print(loss[1])