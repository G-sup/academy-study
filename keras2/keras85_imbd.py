from tensorflow.keras import models
from tensorflow.keras.datasets import reuters, imdb
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding,Dense,Conv1D,LSTM,Flatten, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers.core import Dropout


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)


# 데이터 구조 확인
print(x_train[0], type(x_train[0]))
print(y_train[0])
print(len(x_train[0]), len(x_train[11])) # 87 59
print('==============================================================')
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)

# 실습


x_train = pad_sequences(x_train, maxlen=239, padding='pre')
x_test = pad_sequences(x_test, maxlen=239, padding='pre')
print(x_train.shape, x_test.shape) # (8982,100), (2246, 100)

# 모델
from tensorflow.keras.layers import Embedding,Dense,Conv1D,LSTM,Flatten, BatchNormalization, GlobalAveragePooling1D

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim = 64, input_length= 239))
# model.add(Embedding(1000,64))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Conv1D(32, 2, activation='relu'))
model.add(BatchNormalization())
model.add(GRU(64, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',factor = 0.5, patience= 5)
model.fit( x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es,lr])


results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])

# loss : 1.1470750570297241
# acc : 0.8004400134086609

# loss : 1.1620186567306519
# acc : 0.8136799931526184

# loss : 0.5735373497009277
# acc : 0.8258000016212463


# model.add(GRU(64, activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# loss : 0.44818374514579773
# acc : 0.838919997215271

# model.add(GRU(64, activation= 'relu'))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# loss : 1.0235515832901
# acc : 0.8088399767875671

# model.add(LSTM(64, activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# loss : 0.693152129650116
# acc : 0.5

# model.add(Conv1D(32, 2, activation='relu'))
# model.add(BatchNormalization())
# model.add(GRU(64, activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# loss : 0.9565762281417847
# acc : 0.8489999771118164

# model.add(Conv1D(32, 2, activation='relu'))
# model.add(Conv1D(32, 2, activation='relu'))
# model.add(BatchNormalization())
# model.add(GRU(64, activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# loss : 0.8730303645133972
# acc : 0.8537200093269348

# model.add(Conv1D(32, 2, activation='relu'))
# model.add(Conv1D(32, 2, activation='relu'))
# model.add(BatchNormalization())
# model.add(GRU(64, activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# loss : 1.3097283840179443
# acc : 0.8581200242042542