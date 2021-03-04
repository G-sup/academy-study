from numpy.core.fromnumeric import shape
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()

token.fit_on_texts([text])

print(token.word_index)

x = token.texts_to_sequences([text])

print(x)

# 나는 :3 은 진짜 : 1 의 세배가 아니기 때문에 원핫인코딩을 해줘야 한다
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)

x = to_categorical(x)

print(x)
print(x.shape)