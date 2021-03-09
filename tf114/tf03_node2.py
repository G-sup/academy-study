# +
# -
# *
# /

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
sess = tf.Session()

print('==================덧셈=======================')

node3 = tf.add(node1, node2)
print(node3) # Tensor("Add:0", shape=(), dtype=float32)

print('sess.run([node1,node2]) : ', sess.run([node1,node2]))
print('덧셈 값 : ', sess.run(node3))
print('==================뺄셈=======================')

node4 = tf.subtract(node1, node2)
print(node3) # Tensor("Add:0", shape=(), dtype=float32)

print('sess.run([node1,node2]) : ', sess.run([node1,node2]))
print('뺄셈 값) : ', sess.run(node4))

print('===================곱셈======================')

node5 = tf.multiply(node1, node2)
print(node3) # Tensor("Add:0", shape=(), dtype=float32)

print('sess.run([node1,node2]) : ', sess.run([node1,node2]))
print('곱셈 값 : ', sess.run(node5))


print('==================나눗셈=====================')

node6 = tf.divide(node1, node2)
node7 = tf.mod(node1, node2)
print(node3) # Tensor("Add:0", shape=(), dtype=float32)

print('sess.run([node1,node2]) : ', sess.run([node1,node2]))
print('나눗셈 값 : ', sess.run(node6))
print('나머지 : ', sess.run(node7))

print('=============================================')

