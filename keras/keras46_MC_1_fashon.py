import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

(x_train, y_train), (x_test,  y_test) = fashion_mnist.load_data()


print(x_test.shape) #(10000, 28, 28)
print(x_train.shape) #(60000, 28, 28)
print(y_test.shape)  #(10000,)
print(y_train.shape) #(60000,)
 

# plt.imshow(x_train[0],'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_test = x_test.reshape(-1,28,28,1)
x_train = x_train.reshape(-1,28,28,1)


from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


#2

model = Sequential()
model.add(Conv2D(64,(2,2),padding='same',strides=2,input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(128, (2,2),padding='same', strides=2))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.4))
model.add(Conv2D(128, (2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = './modelCheckPoint/k46_MC_1_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='loss',patience=3,mode='auto')
model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss = model.evaluate(x_test,y_test)
print(loss)

