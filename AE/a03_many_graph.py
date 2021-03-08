import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


def autoencode(hidden_layer_size) :
    model = Sequential()
    model.add((Dense(units=hidden_layer_size,input_shape = (784,),activation='relu')))
    model.add(Dense(units=784,activation='sigmoid'))
    return model

model_01 = autoencode(hidden_layer_size=1)
model_02 = autoencode(hidden_layer_size=2)
model_03 = autoencode(hidden_layer_size=4)
model_04 = autoencode(hidden_layer_size=8)
model_05 = autoencode(hidden_layer_size=16)
model_06 = autoencode(hidden_layer_size=32)

print('node 1개 시작')
model_01.compile(optimizer='adam',loss='binary_crossentropy')
model_01.fit(x_train,x_train,epochs= 10)

print('node 2개 시작')
model_02.compile(optimizer='adam',loss='binary_crossentropy')
model_02.fit(x_train,x_train,epochs= 10)

print('node 4개 시작')
model_03.compile(optimizer='adam',loss='binary_crossentropy')
model_03.fit(x_train,x_train,epochs= 10)

print('node 8개 시작')
model_04.compile(optimizer='adam',loss='binary_crossentropy')
model_04.fit(x_train,x_train,epochs= 10)

print('node 16개 시작')
model_05.compile(optimizer='adam',loss='binary_crossentropy')
model_05.fit(x_train,x_train,epochs= 10)

print('node 32개 시작')
model_06.compile(optimizer='adam',loss='binary_crossentropy')
model_06.fit(x_train,x_train,epochs= 10)


output_1 = model_01.predict(x_test)
output_2 = model_02.predict(x_test)
output_3 = model_03.predict(x_test)
output_4 = model_04.predict(x_test)
output_5 = model_05.predict(x_test)
output_6 = model_06.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7,5,figsize = (15, 15))


random_imgs = random.sample(range(output_1.shape[0]),5)
outputs = [x_test,output_1,output_2,output_3,output_4,output_5,output_6]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28),cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()




