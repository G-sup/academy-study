from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
from tensorflow._api.v1 import train

tf.set_random_seed(104)

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,6,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0,],[0,1,0],[1,0,0],[1,0,0]]

x = tf.compat.v1.placeholder('float',shape=[None,4])
y = tf.compat.v1.placeholder('float',shape=[None,3])

w = tf.compat.v1.Variable(tf.random_normal([4,3],name = 'weight'))
b = tf.compat.v1.Variable(tf.random_normal([1,3]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_data,y:y_data})
        if step % 200 ==0:
            print(step, cos_val)

    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(a,'\n' ,sess.run(tf.argmax(a, 1)),'번째')

