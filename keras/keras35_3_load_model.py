import numpy as np
from tensorflow.keras.models import load_model

#1
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print(dataset)
x = dataset[:, :4] # x = dataset[0:6, 0:4]
y = dataset[:, -1] # y = dataset[0:6, 4], y = dataset[:, 4]

x = x.reshape(6, 4, 1)


#2

model = load_model('./model/save_keras35.h5')
model.summary()

#3
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x, y, epochs=250, batch_size=1, validation_split = 0.2, verbose=1)

#4

loss = model.evaluate(x,y)

print(loss)


# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually
