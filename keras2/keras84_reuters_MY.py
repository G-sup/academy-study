from numpy.core.defchararray import index
from numpy.lib.arraysetops import unique
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=5000, test_split=0.2)


'''
# 데이터 구조 확인
print(x_train[0], type(x_train[0]))
print(y_train[0])
print(len(x_train[0]), len(x_train[11])) # 87 59
print('==============================================================')
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print("뉴스기사 최대길이 : ", max(len(l) for l in x_train)) # 2376
print("뉴스기사 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# y분포
unique_elemnets, count_elements = np.unique(y_train, return_counts=True)
print('y분포 : ', dict(zip(unique_elemnets, count_elements)))  # dict 딕셔너리 형태, zip 합치는것 ex) 0과 55, 1과 432
print('=============================================================')

# plt.hist(y_train, bins = 46)
# plt.show()

# x의 단어 분포
word_to_index = reuters.get_word_index() # keras의 datasets에서만 사용가능
print(word_to_index)
print(type(word_to_index))
print('=============================================================')

#  키와 벨류를 교체
index_to_word = {}

for key, value in word_to_index.items():
    index_to_word[value] = key

#  키와 벨류를 교체후
print(index_to_word)
print(index_to_word[1]) # the
print(len(index_to_word)) # 30979
print(index_to_word[30979]) # northerly

# 글씨 복원
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

'''
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x_train, padding='pre', maxlen = 146) # pre = 앞부분, post = 뒷부분 을 0으로 채워준다, maxlen 최대 길이 조절 
pad_x_test = pad_sequences(x_test, padding='pre', maxlen = 146) # pre = 앞부분, post = 뒷부분 을 0으로 채워준다, maxlen 최대 길이 조절 

print(pad_x)
print(pad_x.shape) # (8982, 146) 

print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 1997

print(pad_x_test)
print(pad_x_test.shape) # (2246, 146)

print(np.unique(pad_x_test)) # [   0    1    2 ... 9995 9996 9999]
print(len(np.unique(pad_x_test))) # 8421

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

print(y_train.shape) # (8982, 46)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, GRU

model = Sequential()

# input_dim= 총 단어사전의 개수, output_dim = 다음레이어로 주는 아웃풋(백터화), input_length= 데이터의 컬럼수
model.add(Embedding(input_dim=5000, output_dim = 64, input_length= 146))
# model.add(Embedding(28,128)) # input_length = None 이 된다 자동으로 데이터 컬럼은 들어가게된다

model.add(GRU(128,activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(46,activation='softmax'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',factor = 0.1, patience= 5)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(pad_x,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=1,callbacks=[es,lr])

acc = model.evaluate(pad_x_test, y_test)
print(acc)

# [2.013153076171875, 0.6767587065696716]