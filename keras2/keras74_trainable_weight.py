import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])



model = Sequential()
model.add(Dense(4,input_dim = 1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
print(model.trainable_weights)

print(len(model.weights)) # layer 하나 bias 하나 해서 길이는 레이어의 두배 ex) 10
print(len(model.trainable_weights)) # 상동 10
