import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
print(dataset.values())

print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

x = dataset.data
# x = dataset['data']
y= dataset.target
# y= dataset['target']

print(x)
print(y)
print(x.shape)         # (150, 4)
print(type(x),type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# numpy를 pandas로
df = pd.DataFrame(x, columns=dataset.feature_names) # header(열의 이름) 설정( 데이터는 아니다 데이터 설명용)
# df = pd.DataFrame(x, columns=dataset['feature_names']) # 위에랑 같은 표현

print(df)

print(df.shape)
print(df.columns) # header(열의 이름)
print(df.index)   # 명시를 해주지않으면 자동으로 index 1~~~끝

print(df.head()) # 디폴트 5 = df[:5]
print(df.tail()) # 디폴트 5 = df[-5:]
print(df.info())
# non - null = 결측치가 없다 (비어있는게 없다)
print(df.describe())

df.columns = ['sepal_length','sepal_width','petal_length','petal_width'] # 열(columns)의 이름변경
print(df.columns) 
print(df.info())
print(df.describe())

# x에 y열을 추가
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())
print(df.shape) # 150,5
print(df.columns)
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull()) # non - null = False  결측치가 없다 (비어있는게 없다)
print(df.isnull().sum()) # non - null = 0  결측치가 없다 (비어있는게 없다)
print(df.describe())
print(df['Target'].value_counts()) # [열이름]몇개씩인지 카운트

# 상관계수
print(df.corr()) # 상관계수 확인

# 시각화

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 도수 분포도

plt.figure(figsize=(10, 6))

plt.subplot(2,2,1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width',data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length',data=df)
plt.title('petal_length')


plt.subplot(2,2,4)
plt.hist(x='petal_width',data=df)
plt.title('petal_width')

plt.show() #  x 은 개수  y는 길이 (이 그래프에서) 즉 분포도를 볼수있다
