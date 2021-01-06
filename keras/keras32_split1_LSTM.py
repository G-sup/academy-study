import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) #[item for item in subset]
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print(dataset)
x = dataset[:, :4] # x = dataset[0:6, 0:4]
y = dataset[:, -1] # y = dataset[0:6, 4], y = dataset[:, 4]


x_pred = dataset[-1,1:]

print(x) #6,4 
print(x.shape) #6,4 
print(y) #6,
print(y.shape) #4,

x = x.reshape(6, 4, 1)



#2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
 
model = Sequential()
model.add(LSTM(356, input_shape=(4,1)))
model.add(Dense(356))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3
model.compile(loss='mse',optimizer='adam',metrics='mse')
model.fit(x, y, epochs=250, batch_size=1, validation_split = 0.2, verbose=1)
#4
loss = model.evaluate(x,y)

print(loss)

x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)

print(y_pred)
