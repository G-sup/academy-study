import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/data/dacon/dacon3/dirty_mnist_2nd_noise_clean/00000.png")

plt.figure(figsize=(15,12))
plt.imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(15,12))
plt.imshow(img_gray)

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
plt.figure(figsize=(15,12))
plt.imshow(img_blur)

ret, img_th = cv2.threshold(img_blur, 100, 230, cv2.THRESH_BINARY_INV)

image, contours, hierachy= cv2.findContours(img_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(each) for each in contours]
rects

tmp = [w*h for (x,y,w,h) in rects]
tmp.sort()
tmp

rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>15000)and(w*h<500000))]
rects

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), 
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 

plt.figure(figsize=(15,12))
plt.imshow(img);