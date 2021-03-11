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

# w1 = tf.compat.v1.Variable(tf.random_normal([784,100], stddev=0.1), name='weight')
# get_variable와 Variable은 거의 비슷
w1 = tf.compat.v1.get_variable('weight1',shape = [784,256],initializer = tf.contrib.layers.variance_scaling_initializer()) 

# sigmoid나 tanh일때 잘먹힐 확률이 높다, xavier_initializer = kernel_initializer
# relu계열에 잘먹힐 확률이 높다, variance_scaling_initializer(he_initializer는 없다) = kernel_initializer 
print('w1 : ', w1)
b1 = tf.compat.v1.Variable(tf.random_normal([256], stddev=0.1), name = 'bias')
print("b1 : ", b1)
# layer1 = tf.matmul(x,w1) + b1
# layer1 = tf.nn.softmax(tf.matmul(x,w1)+b1)
# layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)
# layer1 = tf.nn.selu(tf.matmul(x,w1)+b1)
layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)
print('layer1 : ',layer1)
# ========Drop out 적용법========
layer1 = tf.nn.dropout(layer1, keep_prob=0.7) 
# ==============================
print('layer1 : ',layer1)




# w2 = tf.compat.v1.Variable(tf.random_normal([100,50], stddev=0.1), name='weight2')
# get_variable와 Variable은 거의 비슷
w2 = tf.compat.v1.get_variable('weight2',shape = [256,128],initializer = tf.contrib.layers.variance_scaling_initializer()) 
b2 = tf.compat.v1.Variable(tf.random_normal([128], stddev=0.1), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,w2)+b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.7) 

w3 = tf.compat.v1.get_variable('weight3',shape = [128,64],initializer = tf.contrib.layers.variance_scaling_initializer()) 
b3 = tf.compat.v1.Variable(tf.random_normal([64], stddev=0.1), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,w3)+b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.7) 

w4 = tf.compat.v1.get_variable('weight4',shape = [64,10],initializer = tf.contrib.layers.xavier_initializer()) 
b4 = tf.compat.v1.Variable(tf.random_normal([10], stddev=0.1), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4) # 최종 아웃풋


# 컴파일 훈련
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.00008).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(loss)

trianing_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) #  60000/ 50 = 1200, 60000/ 100 = 600 , 60000/ 200 = 300

sess = tf.Session()
sess.run(tf.global_variables_initializer())

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

# # 결과
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(2001):
#         _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
#         if step % 20 ==0:
#             print(step, cos_val)

#     y_acc_test = sess.run(tf.argmax(y_test, 1))
#     predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
#     acc = accuracy_score(y_acc_test, predict)
#     print("accuracy_score : ", acc)
