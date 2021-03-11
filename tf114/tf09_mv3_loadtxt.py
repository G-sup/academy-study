# 

import tensorflow as tf
import numpy as np

tf.set_random_seed(104)


dataset = np.loadtxt('C:/data/Csv/data-01-test-score.csv',delimiter=',',ndmin=2)

print(dataset.shape)
x_data = dataset[:,:3]
y_data = dataset[:,-1:]

print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32,shape = [None,3])
y = tf.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random_normal([1]),name = 'bias')

# hypothesis = x*w+b
hypothesis = tf.matmul(x, w) + b # matmul = 곱하기

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=(0.00004))
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val,_ = sess.run([cost,hypothesis,train], feed_dict={x:x_data,y:y_data})
    if step % 10 == 0:
        print(step, 'cost : ', cost_val,"\n 예측값 : \n", hy_val)

print('결과 : ', sess.run(hypothesis, feed_dict={x:[[73,66,70]]}))
print('결과 : ', sess.run(hypothesis, feed_dict={x:[[93,88,93]]}))
print('결과 : ', sess.run(hypothesis, feed_dict={x:[[89,91,90]]}))
print('결과 : ', sess.run(hypothesis, feed_dict={x:[[96,98,100]]}))
print('결과 : ', sess.run(hypothesis, feed_dict={x:[[73,66,70]]}))


sess.close()


'''
73,80,75,152 결과 :  [[152.01897]]
93,88,93,185 결과 :  [[185.69945]]
89,91,90,180 결과 :  [[181.30904]]
96,98,100,196 결과 :  [[198.37654]]
73,66,70,142 결과 :  [[141.72066]]
'''
