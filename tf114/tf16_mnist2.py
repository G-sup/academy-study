import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
(x_train, y_train), (x_test,y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(-1,28*28).astype('float32')/255.
x_test = x_test.reshape(-1,28*28).astype('float32')/255.

x = tf.compat.v1.placeholder(dtype='float32',shape = [None,784])
y = tf.compat.v1.placeholder(dtype='float32',shape = [None,10])

w = tf.compat.v1.Variable(tf.random_normal([784,10]),name='weight')
b = tf.compat.v1.Variable(tf.random_normal([10]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b )


# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)
