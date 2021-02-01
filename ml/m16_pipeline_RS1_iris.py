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

# Parameters = [
#     {"mo__C" : [1,10,100,1000],"mo__kernel":["linear"] },
#     {"mo__C" : [1,10,100],"mo__kernel":['rbf'],"mo__gamma":[0.001,0.0001]},
#     {"mo__C" : [1,10,100,1000],"mo__kernel":["sigmoid"],"mo__gamma":[0.001,0.0001]}
# ]

Parameters = [
    {"svc__C" : [1,10,100,1000],"svc__kernel":["linear"] },
    {"svc__C" : [1,10,100],"svc__kernel":['rbf'],"svc__gamma":[0.001,0.0001]},
    {"svc__C" : [1,10,100,1000],"svc__kernel":["sigmoid"],"svc__gamma":[0.001,0.0001]}
]
# 2

# pipe = Pipeline([('scaler',MinMaxScaler()),('mo',SVC())]) 
# 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다  이름을 정해줘야 위에 Parameters를 조정가능하다(mo__:이름으로 지정) 
pipe = make_pipeline(StandardScaler(),SVC()) #이걸 사용할때는 (SVC__)로 해야 된다

# model = GridSearchCV(pipe,Parameters,cv = 5)
model = RandomizedSearchCV(pipe,Parameters,cv = 5)

model.fit(x_train, y_train)

results = model.score(x_test,y_test)

print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 
print(results)

# ===================
# for문 으로 grid랑 random돌릴때
# models = [GridSearchCV(pipe,Parameters,cv = 5),RandomizedSearchCV(pipe,Parameters,cv = 5)]

# for algorithm in models :  
#     model = algorithm
#     model.fit(x_train, y_train)
#     print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 
#     print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다
# ======================

# 0.9666666666666667
# 1.0