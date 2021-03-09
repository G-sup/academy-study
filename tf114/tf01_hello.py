import tensorflow as tf
print(tf.__version__)

hello  = tf.constant("Hello World") # constant = 상수(고정값)  세가지 자료형 중에 하나
print(hello)
# Tensor("Const:0", shape=(), dtype=string) 
# 그냥 출력하면 텐서의 자료형이 들어가 있다 

sess = tf.Session()
# tf.Session().run 을 지나가야지 결과 값이 나온다

print(sess.run(hello)) # run 으로 세션을 동작 시켜야된다
