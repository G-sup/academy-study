#  imagegrnerator의 fit 사용

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input # vgg16에 맞게 맞춰준다

# pred0 = load_img('../data/image/me/1/KakaoTalk_20210210_101432928.jpg',target_size=(100,100))
# pred1 = img_to_array(pred0)
# pred = preprocess_input(pred1)
# pred = pred.reshape(1,100,100,3)

img_1 = load_img('../data/image/me/1/1.jpg',target_size=(100,100))
img_2 = load_img('../data/image/me/1/2.jpg',target_size=(100,100))
img_3 = load_img('../data/image/me/1/3.jpg',target_size=(100,100))
img_11 = load_img('../data/image/me/1/11.jpg',target_size=(100,100))
img_22 = load_img('../data/image/me/1/22.jpg',target_size=(100,100))

arr_1 = img_to_array(img_1)
arr_2 = img_to_array(img_2)
arr_3 = img_to_array(img_3)
arr_11 = img_to_array(img_11)
arr_22 = img_to_array(img_22)

arr_1 = preprocess_input(arr_1)
arr_2 = preprocess_input(arr_2)
arr_3 = preprocess_input(arr_3)
arr_11 = preprocess_input(arr_11)
arr_22 = preprocess_input(arr_22)


arr_input = np.stack([arr_1,arr_2,arr_3,arr_11,arr_22])


x_train = np.load('C:/data/npy/keras67_train_x.npy')
y_train = np.load('C:/data/npy/keras67_train_y.npy')

x_test = np.load('C:/data/npy/keras67_test_x.npy')
y_test = np.load('C:/data/npy/keras67_test_y.npy')

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

vgg16.trainable = False
vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
lr = ReduceLROnPlateau(monitor='val_loss',patience=13,factor=0.5,verbose=1) 
es = EarlyStopping(monitor='val_loss',patience=25,mode='auto')
model.fit(x_train,y_train,verbose=1,batch_size=4,epochs=1000 ,validation_split= 0.2, callbacks=[es,lr])

acc = model.evaluate(x_test,y_test)
print(acc)

pred = model.predict(arr_input)



print(pred)

# print('남자일 확률 : ',"%0.1f%%" %(pred*100))
# print('여자일 확률 : ',"%0.1f%%" %((1-pred)*100))

# pred = np.where(pred>0.5, 1, pred)
# pred = np.where(pred<0.5, 0, pred)

# if pred >= 1:
#     print("남자")
# else:
#     print("여자")