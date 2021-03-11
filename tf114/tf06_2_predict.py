# placeholder 사용

import tensorflow as tf

tf.set_random_seed(104)


# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])


W = tf.Variable(tf.random_normal([1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='bias')
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(B))


hypothesis = x_train * W + B

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss(cost) = mse 와 같다  square(제곱), reduce(평균)
# print(sess.run(cost))


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost) # optimizer에 cost를 넣고 train 시키겠다 = 최적의 weight 가 나올수 있게

# 위의 코드 두줄과 같다
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(tf.reduce_mean(tf.square(x_train * W + B - y_train)))


# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # (변수의) 마지막 부분에 한번만 넣어주면 된다
# for step in range(2001):
#     sess.run(train)
#     if step % 20 ==0: # 20번 마나 출력
#         print(step,sess.run(cost), sess.run(W), sess.run(B)) # epochs, loss, weight, bias 와 같다
# sess.close()

# 위의 코드와 같다
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # (변수의) 마지막 부분에 한번만 넣어주면 된다

    for step in range(2001):
        # sess.run(train)
        _, cost_val, W_val, B_val = sess.run([train,cost,W,B], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})

        if step % 20 ==0: # 20번 마나 출력
            # print(step,sess.run(cost, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
            # , sess.run(W, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
            # , sess.run(B, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})) # epochs, loss, weight, bias 와 같다
            print(step, cost_val, W_val, B_val)
    # predict 값을 도출
    print(sess.run(hypothesis, feed_dict={x_train: [6,7,8]}))
            


# 1. [4]
# 2. [5,6]
# 3. [6,7,8]