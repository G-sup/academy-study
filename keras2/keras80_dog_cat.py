# 이미지 루트 data/image/vgg 
# 개 고양이 라이언 슈트

from nose import result
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1

# 이미지 로드
img_dog = load_img('../data/image/vgg/dog1.jpg',target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg',target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.jpg',target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg',target_size=(224,224))

# plt.imshow(img_dog)
# plt.show()

# print(img_dog) # <PIL.Image.Image image mode=RGB size=224x224 at 0x2B3FE948070>

# 이미지 수치화
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(arr_dog)
print(type(arr_dog)) # <class 'numpy.ndarray'>
print(arr_dog.shape) # (224, 224, 3)

# RGB -> BGR(open cv)

from tensorflow.keras.applications.vgg16 import preprocess_input # vgg16에 맞게 맞춰준다

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)

print(arr_dog)
print(arr_dog.shape)

# (1, 224, 224, 3) 로 만들고 네개를 합쳐서 (4, 224, 224, 3) 를 만든다

arr_input = np.stack([arr_dog,arr_cat,arr_lion,arr_suit])
print(arr_input.shape)

# 2
model = VGG16()
result = model.predict(arr_input)

print(result)
print('results.shape : ', result.shape) # results.shape :  (4, 1000) 
# 이미지넷에서 분류할 수 있는 카테고리 개수 = 1000개

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

results = decode_predictions(result)

print('==================================================================')
print("results[0] : ", results[0])
print('==================================================================')
print("results[1] : ", results[1])
print('==================================================================')
print("results[2] : ", results[2])
print('==================================================================')
print("results[3] : ", results[3])
print('==================================================================')
