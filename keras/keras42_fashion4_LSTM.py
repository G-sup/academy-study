import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM, GRU
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

x_test = x_test.reshape(-1,28,28)
x_train = x_train.reshape(-1,28,28)


from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


#2

model = Sequential()
model.add(GRU(50,activation='relu',input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
early_stopping = EarlyStopping(monitor='loss',patience=3,mode='auto')
model.fit(x_train, y_train, epochs=40,batch_size=34,validation_split=0.2,verbose=1,callbacks=[early_stopping])

#4
loss = model.evaluate(x_test,y_test)
print(loss)
