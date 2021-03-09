# from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) #False
# 즉시 실행 모드
tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 끈다

print(tf.executing_eagerly()) #False


print(tf.__version__)


hello  = tf.constant("Hello World") # constant = 상수(고정값)  세가지 자료형 중에 하나
print(hello)
# Tensor("Const:0", shape=(), dtype=string) 
# 그냥 출력하면 텐서의 자료형이 들어가 있다 

# sess = tf.Session()
sess = tf.compat.v1.Session() # 1.14

print(sess.run(hello)) # run 으로 세션을 동작 시켜야된다
