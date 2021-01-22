import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K




# quantile loss

def quantile_loss(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
# mean = 평균값


# #  = mse
# def custom_mse(y_true,y_pred):
#     return tf.math.reduce_mean(tf.square(y_true - y_pred))

# # y_true = y 원래값, y_pred = y 예측값


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
model.compile(loss=quantile_loss,optimizer='adam')
model.fit(x,y,epochs=30,batch_size =1)

# 4
loss = model.evaluate(x,y)
print(loss)