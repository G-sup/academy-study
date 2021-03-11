import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [3., 5., 7.]

W =tf.compat.v1.placeholder(tf.float32)

hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_hist = []
cost_hist = []

with tf.compat.v1.Session() as sess :
    for i in range(-30,50):
        curr_w = i# * 0.1
        curr_cost = sess.run(cost, feed_dict = {W:curr_w}) 

        w_hist.append(curr_w)
        cost_hist.append(curr_cost)

print("=================================================")
print(w_hist)
print("=================================================")
print(cost_hist)
print("=================================================")

plt.plot(w_hist,cost_hist)
plt.show()