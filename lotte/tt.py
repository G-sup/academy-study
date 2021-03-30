import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,zoom_range=1.2,shear_range=0.7,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,shear_range=0.3,validation_split = 0.2,fill_mode='nearest')
test_datagen = ImageDataGenerator()

xy_train = train_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',seed=104,target_size=(128, 128),batch_size=100000, subset="training")
xy_train = train_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',target_size=(128,128),batch_size=50000,seed=42,subset='training',class_mode='categorical')  

xy_test = test_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',seed=104,target_size=(128, 128),batch_size=100000 , subset="validation")
xy_test = train_datagen.flow_from_directory('C:/data/lotte/LPD_competition/train',target_size=(128,128),batch_size=50000,seed=42,subset='validation',class_mode='categorical')  

x_pred = test_datagen.flow_from_directory('C:/data/lotte/LPD_competition/test',target_size=(128, 128),batch_size=72000)
x_pred = test_datagen.flow_from_directory('C:/data/lotte/LPD_competition/test',target_size=(128,128),batch_size=72000,seed=42,class_mode=None)   

np.save('C:/data/lotte/npy/train_x.npy', arr=xy_train[0][0])
np.save('C:/data/lotte/npy/train_y.npy', arr=xy_train[0][1])
np.save('C:/data/lotte/npy/test_x.npy', arr=xy_test[0][0])
np.save('C:/data/lotte/npy/test_y.npy', arr=xy_test[0][1])
np.save('C:/data/lotte/npy/pred_x.npy', arr=x_pred[0][0])

np.save('../data/LPD_competition/npy/data_x_train1.npy', arr=xy_train[0][0], allow_pickle=True)
np.save('../data/LPD_competition/npy/data_y_train1.npy', arr=xy_train[0][1], allow_pickle=True)
np.save('../data/LPD_competition/npy/data_x_val1.npy', arr=xy_val[0][0], allow_pickle=True)
np.save('../data/LPD_competition/npy/data_y_val1.npy', arr=xy_val[0][1], allow_pickle=True)
np.save('../data/LPD_competition/npy/data_x_pred1.npy', arr=x_pred[0], allow_pickle=True)
