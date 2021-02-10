# 나를 찍어서 남자인지 여자인지  

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

pred_datagen = ImageDataGenerator(rescale=1./255)

pred = pred_datagen.flow_from_directory('C:/data/image/me',target_size=(100, 100),batch_size=1 ,class_mode='binary')

model = load_model("../Data/h5/save_keras67_4.h5")
pred = model.predict(pred)

print('남자일 확률 : ',"%0.1f%%" %(pred*100))
print('여자일 확률 : ',"%0.1f%%" %((1-pred)*100))

pred = np.where(pred>0.5, 1, pred)
pred = np.where(pred<0.5, 0, pred)

if pred >= 1:
    print("남자")
else:
    print("여자")

