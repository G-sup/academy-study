import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf





#  = mse
def custom_mse(y_true,y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
# y_true = y 원래값, y_pred = y 예측값









x = np.array([1,2,3,4,5,6,7,8]).astype('float32') # int형이 안먹는다.
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

print(x.shape)




# 2
model = Sequential()
model.add(Dense(10, input_shape = (1, )))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3
model.compile(loss=custom_mse,optimizer='adam')
model.fit(x,y,epochs=30,batch_size =1)

# 4
loss = model.evaluate(x,y)
print(loss)