import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/data/dacon/dacon3/dirty_mnist_2nd/00042.png')
dst = cv2.fastNlMeansDenoising(img,None,45,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()

