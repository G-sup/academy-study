import tensorflow as tf
import numpy as np
from tensorflow.lite.python.lite import Optimize
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.random_ops import categorical
tf.set_random_seed(104)

# 1 
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

learning_rate = 0.01
trianing_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(dtype='float32', shape=[None,28,28,1])
y = tf.compat.v1.placeholder(dtype='float32', shape=[None,10])

# 2

# L1.
w1 =  tf.compat.v1.get_variable("w1",shape=[3,3,1,128])
L1 = tf.nn.conv2d(x,w1, strides=[1,1,1,1],padding='SAME')
print(L1)
# Conv2D(filter, kernel_size, input_shape)
# Conv2D(32, (3,3), input_shape=(28, 28, 1)) 
# strides=[1,2,2,1] : strides=2 일떄  
# strides=[1,3,3,1] : strides=3 일떄 앞뒤에 1은 차원을 맞춰주는 용도
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') # padding='SAME'은 줄일때 홀수이면 +1 VALID는 -1
print(L1)

# L2.
w2 =  tf.compat.v1.get_variable("w2",shape=[3,3,128,64])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') 
print(L2)

# L3.
w3 =  tf.compat.v1.get_variable("w3",shape=[3,3,64,32])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') # padding='SAME' :(?, 4, 4, 8) , padding='VALID' :(?, 3, 3, 8)
print(L3)

# L4.
w4 =  tf.compat.v1.get_variable("w4",shape=[3,3,32,16])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1],padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') # padding='SAME' :(?, 4, 4, 8)
print(L4)

# Flatten
L_flat = tf.reshape(L4, [-1,L4.shape[1]*L4.shape[2]*L4.shape[3]])
print(L_flat)

# L5
w5 = tf.compat.v1.get_variable('w5', shape = [L4.shape[1]*L4.shape[2]*L4.shape[3], 8],initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.compat.v1.Variable(tf.random.normal([8]), name='b5')
L5 = tf.nn.relu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, rate=0.2)
print(L5)

# L6
w6 = tf.compat.v1.get_variable('w6', shape = [8, 4],initializer=tf.contrib.layers.variance_scaling_initializer())
b6 = tf.compat.v1.Variable(tf.random.normal([4]), name='b6')
L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)
L6 = tf.nn.dropout(L6, rate=0.2)
print(L6)

# hypothesis
w7 = tf.compat.v1.get_variable('w7', shape = [4, 10],initializer=tf.contrib.layers.variance_scaling_initializer())
b7 = tf.compat.v1.Variable(tf.random.normal([10]), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print(hypothesis)

#3

loss =  tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) #categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 훈련
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(trianing_epochs):
    avg_cost = 0

    for i in range(total_batch):     # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('epoch : ','%04d' %(epoch+1),'cost = {:.9f}'.format(avg_cost))

print(" 훈련 끝 ")

prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print('acc : ', sess.run(accuracy,feed_dict = {x:x_test,y:y_test}))