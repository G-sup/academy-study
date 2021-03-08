#  imagegrnerator의 fit_generator 사용

from token import LEFTSHIFT
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

# train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,
#     rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

# # image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    

# train = train_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=100 ,class_mode='binary', subset="training")
# test = test_datagen.flow_from_directory('C:/data/image/data2',seed=104,target_size=(100, 100),batch_size=86 ,class_mode='binary', subset="validation")

# # np.save('C:/data/npy/keras67_train_x.npy', arr=xy_train[0][0])
# # np.save('C:/data/npy/keras67_train_y.npy', arr=xy_train[0][1])
# # np.save('C:/data/npy/keras67_test_x.npy', arr=xy_test[0][0])
# # np.save('C:/data/npy/keras67_test_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/npy/keras67_train_x.npy')

x_test = np.load('C:/data/npy/keras67_test_x.npy')

print(x_train.shape, x_test.shape)

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

# x_test = x_test.reshape(-1,3)
# x_test_noised = x_test_noised.reshape(-1,3)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,BatchNormalization,Conv1DTranspose, LeakyReLU

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add((Conv2D(filters=308,kernel_size=1,input_shape = (100,100,3))))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256,kernel_size=4))    
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=192,kernel_size=2))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=hidden_layer_size,kernel_size=2, activation='tanh'))   
    model.add(Conv2DTranspose(filters=192,kernel_size=2))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters=256,kernel_size=2))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters=308,kernel_size=4,activation='relu'))       
    model.add(Dropout(0.4))  
    # model.add(Flatten())
    model.add(Dense(units=3,activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train,x_train_noised,epochs = 100)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,7))

random_imeges = random.sample(range(output.shape[0]),5)

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_imeges[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_imeges[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('NOISED', size = 20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_imeges[i]].reshape(100,100,3))
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()