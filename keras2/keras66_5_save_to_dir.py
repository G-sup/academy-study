
# ImageDataGenerator 이미지 증폭의 대한 설명


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
    zoom_range=1.2,shear_range=0.7,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory(폴더) 에서 변환

# train_generator
xy_train = train_datagen.flow_from_directory('C:/data/image/brain/train',target_size=(150, 150),batch_size=5 ,class_mode='binary'
    , save_to_dir='C:/data/image/brain_generator/train')
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory('C:/data/image/brain/test',target_size=(150, 150),batch_size=5,class_mode='binary'
    , save_to_dir='C:/data/image/brain_generator/test')
# Found 120 images belonging to 2 classes.


print(xy_train[0][0])  # 정의를 해놓것에 한번 액션을 취해줘야 생성된다
print(xy_train[0][1])  
print(xy_train[0][1].shape) 
# 액션을 취한만큼 생성이 된다 3*5 = 15
