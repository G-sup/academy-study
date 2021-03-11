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


# 모델
x = tf.compat.v1.placeholder(dtype='float32',shape = [None,784])
y = tf.compat.v1.placeholder(dtype='float32',shape = [None,10])

w1 = tf.compat.v1.Variable(tf.random_normal([784,100], stddev=0.1), name='weight')
b1 = tf.compat.v1.Variable(tf.random_normal([100], stddev=0.1), name = 'bias')
# layer1 = tf.matmul(x,w1) + b1
layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # Drop out 적용법

w2 = tf.compat.v1.Variable(tf.random_normal([100,50], stddev=0.1), name='weight2')
b2 = tf.compat.v1.Variable(tf.random_normal([50], stddev=0.1), name = 'bias2')
# layer2 = tf.matmul(layer1,w2) + b2

# ===========relu 사용법====================
# layer2 = tf.nn.selu(tf.matmul(x,w2)+b2)
# layer2 = tf.nn.elu(tf.matmul(x,w2)+b2)
layer2 = tf.nn.relu(tf.matmul(layer1,w2)+b2)
# =========================================
layer2 = tf.nn.dropout(layer2, keep_prob=0.3) # Drop out 적용법

w3 = tf.compat.v1.Variable(tf.random_normal([50,10], stddev=0.1), name='weight3')
b3 = tf.compat.v1.Variable(tf.random_normal([10], stddev=0.1), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3) # 최종 아웃풋

# 컴파일 훈련
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 결과
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 20 ==0:
            print(step, cos_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)
