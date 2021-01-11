import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# datasets = fashion_mnist

# (x_train, x_test, y_train, y_test) = d

(x_train, y_train), (x_test,  y_test) = fashion_mnist.load_data()


print(x_test.shape) #(10000, 28, 28)
print(x_train.shape) #(60000, 28, 28)
print(y_test.shape)  #(10000,)
print(y_train.shape) #(60000,)
 

plt.imshow(x_train[0],'gray')
# plt.imshow(x_train[0])
plt.show()
