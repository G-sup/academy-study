import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , Flatten ,MaxPool2D, LSTM, GRU
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

(x_train, y_train), (x_test,  y_test) = cifar100.load_data()


print(x_test.shape) #(10000, 32, 32, 3)
print(x_train.shape) #(50000, 32, 32, 3)
print(y_test.shape)  #(10000, 1)
print(y_train.shape) #(50000, 1)
 

plt.imshow(x_train[0],'gray')
# plt.imshow(x_train[0])
plt.show()
