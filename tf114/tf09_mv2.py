import tensorflow as tf

tf.set_random_seed(104)

x_data = [[73,51,65], [92,98,11], [89,31,33], [99,33,100], [19,66,79]]
y_data = [[152], [185], [180], [205], [142]]

x = tf.placeholder(tf.float32,shape = [None,3])
y = tf.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random_normal([1]),name = 'bias')

# hypothesis = x*w+b
hypothesis = tf.matmul(x, w) + b # matmul = 곱하기

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=(0.000081))
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val,_ = sess.run([cost,hypothesis,train], feed_dict={x:x_data,y:y_data})
    if step % 10 == 0:
        print(step, 'cost : ', cost_val,"\n 예측값 : \n", hy_val)

sess.close()