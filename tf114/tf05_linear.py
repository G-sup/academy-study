
'''
import tensorflow as tf
sess = tf.Session()

tf.set_random_seed(104)


x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='bias')

# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(B))


hypothesis = x_train * W + B

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss(cost) = mse 와 같다  square(제곱), reduce(평균)
# print(sess.run(cost))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost) # optimizer에 cost를 넣고 train 시키겠다 = 최적의 weight 가 나올수 있게

sess.run(tf.global_variables_initializer()) # (변수의) 마지막 부분에 한번만 넣어주면 된다

for step in range(4):
    sess.run(train)
    if step % 1 ==0: # 20번 마나 출력
        print(step,sess.run(cost), sess.run(W), sess.run(B)) # epochs, loss, weight, bias 와 같다
'''


import tensorflow as tf

tf.set_random_seed(66)
x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# print(sess.run(W),sess.run(b))

hypothesis = x_train*W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step<=2:
        print("epoch : ",step)
        print("x : ",x_train)
        print("W : ",sess.run(W))
        print("b : ",sess.run(b))
        print("W*x + b = hypothesis , ",sess.run(hypothesis))
        print("y : ",y_train)
        print("hypothesis - y_train : ",sess.run(hypothesis - y_train))
        print("cost : ",sess.run(cost))
        print("\n\n")