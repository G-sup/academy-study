import keras_preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.backend import conv1d

docs = ['너무 재밌어요','참 최고에요', ' 참 잘 만든 영화에요','추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요','별로에요', '생각보다 지루해요','연기가 어색해요',\
    '재미없어요','너무 재미없다','참 재밌네요','규현이가 잘 생기긴 했어요']



# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

# 0 으로 앞부분 채우기 앞부분 채우는 이유는 뒤에 채우면 0으로 수렴하기 때문

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen = 5) # pre = 앞부분, post = 뒷부분 을 0으로 채워준다, maxlen 최대 길이 조절 

print(pad_x)
print(pad_x.shape) # (13,5) 

print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 28 (0 부터 27 까지인데 maxlen= 4 일때 maxlen으로 인해 11이 잘렸다)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

'''
Embedding layer
원핫인코딩의 문제점은 데이터가 너무 커진다. + 단어와의 거리에 대한 연산(백터화 한다)을 위해서 keras 에선 Embedding 에서 해결 
'''
#################################
# Embedding layer 뺴고 모델 구성 #
#################################

pad_x = pad_x.reshape(-1,5,1)

model = Sequential()
# input_dim= 총 단어사전의 개수, output_dim = 다음레이어로 주는 아웃풋(백터화), input_length= 데이터의 컬럼수
# model.add(Embedding(input_dim=28, output_dim = 128, input_length= 5))
# model.add(Embedding(28,128)) # input_length = None 이 된다 자동으로 데이터 컬럼은 들어가게된다
# model.add(Dense(128,activation='relu',input_shape = (5,)))
model.add(LSTM(128,activation='relu',input_shape = (5,1)))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics='acc')
model.fit(pad_x, labels, epochs = 150)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
