from tensorflow.keras.models import load_model
from tensorflow_addons.layers import GroupNormalization
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import os
import natsort

path = natsort.natsorted(glob.glob("D:/1.animegan_re/animegan2-pytorch-main/samples/results/640,360/*.jpg"))
images = []
for img in path:
    n = cv.imread(img)
    images.append(n)
print(np.array(images).shape)
# print(images.shape)
images = np.array(images)

model = load_model('./sr_model.h5')
model.summary()

for i in range(1,6697):
    g_image = (model.predict(images[i-1:i])+1)*127.5
    g_image = g_image.reshape(720,1280,3)
    print(g_image.shape)
    # other things you need to do snipped
    cv2.imwrite(f'D:/1.animegan_re/animegan2-pytorch-main/samples/640,360/image_{i}.jpg',g_image)


print(g_image.shape)
