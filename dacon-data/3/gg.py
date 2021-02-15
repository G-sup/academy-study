import tensorflow as tf 
# tf.test.is_gpu_available()


# tf.test.gpu_device_name() # 결과로 나오는 GPU는 본인 pc 설정에 따라 다를 수 있습니다.
# '/device:GPU:0'

# tf.debugging.set_log_device_placement(True)

# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)

import tensorflow as tf 
tf.__version__

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

#  pip install tensorflow-gpu==2.3.1