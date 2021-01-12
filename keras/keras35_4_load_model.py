import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

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

model = load_model('../Data/h5/save_keras35.h5')
# model.add(Dense(5)) = 이름 : dense
# model.add(Dense(1)) = 이름 : dense_1
# model.summary()

# 저장되어있는 모델과 이름이 중복이 되어서 에러가 난다


model.add(Dense(5, name='yaho'))
model.add(Dense(1, name="yaho2"))

model.summary()

#3
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x, y, epochs=250, batch_size=1, validation_split = 0.2, verbose=1)

#4

loss = model.evaluate(x,y)

print(loss)


# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually

# ValueError: All layers added to a Sequential model should have unique names. Name "dense" is already the name of a layer in this model. Update the `name` argument to pass a unique name.
# ValueError: All layers added to a Sequential model should have unique names. Name "dense" is already the name of a layer in this model. Update the `name` argument to pass a unique name.
