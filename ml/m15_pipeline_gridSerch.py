# pipeline 전처리와 모델을 합치는것을 말한다

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV ,RandomizedSearchCV # RandomizedSearch 모델셀렉션에 있다
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC

from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


# 1 
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)


# pipe 라인에서 스케일을 해줘서 아레가 필요없다
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# 2
# model = Pipeline([('scaler',MinMaxScaler()),('model',SVC())]) # (pipe) 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다
model = make_pipeline(StandardScaler(),SVC())


model.fit(x_train, y_train)

results = model.score(x_test,y_test)
print(results)

# models = [make_pipeline(MinMaxScaler(),SVC()),make_pipeline(StandardScaler(),SVC())]

# for algorithm in models :  
#     model = algorithm
#     model.fit(x_train, y_train)

#     results = model.score(x_test,y_test)
#     print(results)

# 0.9666666666666667
# 1.0