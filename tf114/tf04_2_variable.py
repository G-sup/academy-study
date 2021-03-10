import tensorflow as tf

sess = tf.Session()


x= tf.Variable([2], dtype=tf.float32, name = 'test')  # Variable 변수 가지 자료형 중에 하나. 연산의 개념

init = tf.global_variables_initializer() # Variable 은 sess.run을 통과전에 무조건 초기화 해야한다 
# 값이 바뀌는건 아니다 한번만 선언 해주면된다(텐서플로우에 사용하기위해)

sess.run(init)

print(sess.run(x))