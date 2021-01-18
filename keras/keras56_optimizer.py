import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.T
y = y.T

#2 
model = Sequential()
model.add(Dense(100,input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop,SGD,Nadam

# optimizer =  Adam(lr=0.1)
# loss :  0.0003261808888055384 predict :  [0.0, 0.0]

# optimizer =  Adam(lr=0.01)
# loss :  2.29221090893017e-12 predict :  [0.0, 0.0]

optimizer =  Adam(lr=0.001) # Adam의 디폴트
# loss :  1.8236078879602102e-12 predict :  [0.0, 0.0]

# optimizer =  Adam(lr=0.0001)
# loss :  0.00017816909530665725 predict :  [0.0, 0.0] # epochs 가 부족함



# optimizer =  Adadelta(lr=0.1)
# loss :  0.00011126128083560616 predict :  [0.0, 0.0]

# optimizer =  Adadelta(lr=0.01)
# loss :  0.00011617808195296675 predict :  [0.0, 0.0]

# optimizer =  Adadelta(lr=0.001)
# loss :  18.51386070251465 predict :  [0.0, 0.0]

# optimizer =  Adadelta(lr=0.0001)
# loss :  63.443931579589844 predict :  [0.0, 0.0]


# optimizer =   Adamax(lr=0.1)
# loss :  0.16755735874176025 predict :  [0.0, 0.0]

# optimizer =   Adamax(lr=0.01)
# loss :  7.162270993302244e-13 predict :  [0.0, 0.0]

# optimizer =   Adamax(lr=0.001)
# loss :  3.296306431366247e-06 predict :  [0.0, 0.0]

# optimizer =   Adamax(lr=0.0001)
# loss :  0.0002504957956261933 predict :  [0.0, 0.0]


# optimizer =   Adagrad(lr=0.1)
# loss :  loss :  152.8744659423828 predict :  [0.0, 0.0]

# optimizer =   Adagrad(lr=0.01)
# loss :  1.830604787755874e-06 predict :  [0.0, 0.0]

# optimizer =   Adagrad(lr=0.001)
# loss :  4.19104217144195e-05 predict :  [0.0, 0.0]

# optimizer =   Adagrad(lr=0.0001)
# loss :  0.003302236320450902 predict :  [0.0, 0.0]


# optimizer =  RMSprop(lr=0.1)
# loss :  6829783.0 predict :  [0.0, 0.0]
# optimizer =   RMSprop(lr=0.01)
# loss :  1.813860535621643 predict :  [0.0, 0.0]

# optimizer =  RMSprop(lr=0.001)
# loss :  0.1603800654411316 predict :  [0.0, 0.0]

# optimizer =   RMSprop(lr=0.0001)
#  5.02258371852804e-05 predict :  [0.0, 0.0]


# optimizer =  SGD(lr=0.1)
# loss :  6829783.0 predict :  [0.0, 0.0]

# optimizer =   SGD(lr=0.01)
# loss :  1.813860535621643 predict :  [0.0, 0.0]

# optimizer = SGD(lr=0.001)
# loss :  [1.7259441165151657e-07, 1.7259441165151657e-07] result :  [[10.999361]]

# optimizer =  SGD(lr=0.0001)
# loss :  [0.0010589599842205644, 0.0010589599842205644] result :  [[10.957643]]


# optimizer =  Nadam(lr=0.1)
# loss :  6829783.0 predict :  [0.0, 0.0]

# optimizer =   Nadam(lr=0.01)
# loss :  1.813860535621643 predict :  [0.0, 0.0]

# optimizer = Nadam(lr=0.001)
# loss :  0.1603800654411316 predict :  [0.0, 0.0]

# optimizer =  Nadam(lr=0.0001)
#  5.02258371852804e-05 predict :  [0.0, 0.0]


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4
loss = model.evaluate(x, y, batch_size=1)
result = model.predict([11])
print('loss : ', loss, "result : ", result)
