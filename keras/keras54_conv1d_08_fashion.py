import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

x_train = np.load('../data/npy/fmnist_x_train.npy') 
x_test = np.load('../data/npy/fmnist_x_test.npy') 
y_train = np.load('../data/npy/fmnist_y_train.npy') 
y_test = np.load('../data/npy/fmnist_y_test.npy') 

print(x_test.shape) #(10000, 28, 28)
print(x_train.shape) #(60000, 28, 28)
print(y_test.shape)  #(10000,)
print(y_train.shape) #(60000,)
 

# plt.imshow(x_train[0],'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_test = x_test.reshape(-1,784,1)
x_train = x_train.reshape(-1,784,1)


from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


#2

model = Sequential()
model.add(Conv1D(64,4,padding='same',strides=2,input_shape=(784,1)))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.4))
model.add(Conv1D(356, 4,padding='same', strides=2))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.4))
model.add(Conv1D(356, 3))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.4))
model.add(Conv1D(128, 3,padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
from tensorflow.keras.callbacks import ModelCheckpoint # callbacks 안에 넣어준다
modelpath = '../data/modelCheckPoint/k46_MC_1_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일명 : 모델명 에포-발리데이션
mc = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='loss',patience=3,mode='auto')
model.fit(x_train, y_train, epochs=50,batch_size=16,validation_split=0.2,verbose=1,callbacks=[early_stopping,mc])

#4
loss = model.evaluate(x_test,y_test)
print(loss)



# [0.8340162038803101, 0.7872999906539917]
# 0.12064923346042633
# 0.9696000218391418