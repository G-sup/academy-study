import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) #[item for item in subset]
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print(dataset.shape)

x = dataset[:,:4]
y = dataset[:,-1]
print(x.shape, y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape)

#2

model = load_model('../Data/h5/save_keras35.h5')
model.add(Dense(32, name='yaho'))
model.add(Dense(1, name="yaho2"))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=15,mode='auto')

#3
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
hist = model.fit(x, y, epochs=1000, batch_size=8, validation_split = 0.2, callbacks=[es])

print(hist)
print(hist.history.keys()) # 'loss', 'acc', 'val_loss', 'val_acc'

print(hist.history['loss'])

# 그래프

import matplotlib.pyplot as plt


# plt.plot(x, y) 와  plt.show() 만 해도 돌아간다
# (x,y)에서 한항목만 넣으면 y자리에 넣어서 순서대로 나온다

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss' , 'train acc', 'val acc'])
plt.show()

# loss 와 val loss 는 가까워야 신뢰성이 높은 모델 , 서로 거리가 멀면 신뢰성이 떨어진다 (과적합)
# loss 는 val loss 보다 무조건 낮게(좋게) 나온다 val loss가 낮으면(좋으면) 신뢰성이 떨어진다
