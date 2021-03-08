import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255
x_test = x_test.reshape(10000, 28,28,1)/255.


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,Conv2D,Flatten,Conv2DTranspose

# def autoencode(hidden_layer_size) :
#     model = Sequential()
    # model.add((Conv2D(units=154,kernel_size=2,input_shape = (28,28,1),activation='relu')))
#     model.add(Dense(units=784,activation='sigmoid'))
#     return model

def autoencode(hidden_layer_size) :
    model = Sequential()
    model.add((Conv2D(filters=256,kernel_size=1,input_shape = (28,28,1),activation='relu')))
    model.add(Conv2D(filters=64,kernel_size=2,activation='relu'))
    model.add(Conv2D(filters=32,kernel_size=2,activation='relu'))
    model.add(Dense(units=16,activation='relu'))          
    model.add(Dense(units=hidden_layer_size,activation='relu'))   
    # model.add(Flatten())
    model.add(Dense(units=16,activation='relu'))
    model.add(Dense(units=32,activation='relu'))
    model.add(Conv2DTranspose(filters=64,kernel_size=2,activation='relu'))
    model.add(Conv2DTranspose(filters=256,kernel_size=2,activation='relu'))             
    model.add(Dense(units=1,activation='sigmoid'))
    return model

model = autoencode(hidden_layer_size=8)

model.summary()

model.compile(optimizer='adam',loss= 'binary_crossentropy',metrics='acc')

model.fit(x_train,x_train,epochs=5)

output = model.predict(x_test)


# 랜덤한 이미지 출력

from matplotlib import pyplot as plt
import random
fig,((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10,)) = plt.subplots(2,5,figsize = (20,7))

random_images = random.sample(range(output.shape[0]),5)

# 원본 이미지
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('INPUT',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()