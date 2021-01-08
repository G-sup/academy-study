
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, padding='same' ,input_shape=(10,10,1)))
# strides=1 = (1,1), strides=2 (2,2)
# padding='same' : row 랑 cols 값을 그대로 ,padding='vaild' : 디폴트값

model.add(MaxPooling2D(pool_size=(2,3))) 
# pool_size=2 = (2,2) 로 3 =(3,3)

model.add(Conv2D(9, (2,2) , padding='valid' ))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2))                                           
model.add(Flatten()) # 아웃풋 낼떄 사용해야 한다
model.add(Dense(1))

model.summary()
