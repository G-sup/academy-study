import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
# x = dataset['data']
y= dataset.target
# y= dataset['target']

# numpy를 pandas로
df = pd.DataFrame(x, columns=dataset.feature_names) # header(열의 이름) 설정( 데이터는 아니다 데이터 설명용)
# df = pd.DataFrame(x, columns=dataset['feature_names']) # 위에랑 같은 표현

df.columns = ['sepal_length','sepal_width','petal_length','petal_width'] # 열(columns)의 이름변경

# x에 y열을 추가

df['Target'] = y




# csv로 저장 to_csv

df.to_csv('../data/csv/iris_sklearn.csv', sep=',') 
# sep = ',' : separate , 로 구분하겠다
