from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train,y_test = train_test_split(x_data,y_data,train_size=0.7,random_state=104)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([10,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random_normal([1]),name = 'bias')

# hypothesis = x*w+b
hypothesis = tf.matmul(x, w) + b # matmul = 곱하기

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=(0.8))
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val,_ = sess.run([cost,hypothesis,train], feed_dict={x:x_train,y:y_train})
    if step % 10 == 0:
        print(step, 'cost : ', cost_val)#,"\n 예측값 : \n", hy_val)
        r2 = r2_score(y_test, sess.run(hypothesis, feed_dict={x: x_test}))
        print('R2: ', r2)

sess.close()

# R2:  0.48081276625410874