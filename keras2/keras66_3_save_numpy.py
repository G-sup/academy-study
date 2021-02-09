
# ImageDataGenerator 에서 x_trian, y_train 분리

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
    zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory(폴더) 에서 변환

# train_generator
xy_train = train_datagen.flow_from_directory('C:/data/image/brain/train',target_size=(150, 150),batch_size=2000 ,class_mode='binary')
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory('C:/data/image/brain/test',target_size=(150, 150),batch_size=2000,class_mode='binary')
# Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000024B42158550>

# 전체 이미지 개수 나누기 배치사이즈가 크기가 된다
# ex) 배치사이즈 5 = (5, 150, 150, 3) * 0 ~ 31
# ex) 배치사이즈 10 = (10, 150, 150, 3) * 0 ~ 15
# ex) 배치사이즈 160 = (160, 150, 150, 3) * 0
# print(xy_train[0])
# print(xy_train[0][0].shape) #(160, 150, 150, 3)
# print(xy_train[0][1])
# print(xy_train[0][1].shape) #(160,)

np.save('C:/data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('C:/data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('C:/data/image/brain/npy/keras66_test_x.npy', arr=xy_test[0][0])
np.save('C:/data/image/brain/npy/keras66_test_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('C:/data/image/brain/npy/keras66_train_y.npy')

x_test = np.load('C:/data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('C:/data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

