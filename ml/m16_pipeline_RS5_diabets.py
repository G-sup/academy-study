# Regressor = 회귀모델 EX) model = ~~~~~~Regressor 단 LogisticRegression는 분류모델


import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import datetime

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=104)

#2 모델 구성

Parameters = [
    {'mo__n_estimators':[100,200],'mo__max_depth' : [10,12],'mo__n_jobs' : [-1]},
    {'mo__n_estimators':[100,200],'mo__max_depth' : [6,8,10,12]},
    {'mo__min_samples_leaf' : [3,5,7,10],'mo__min_samples_split': [2,3,5,10]},
    {'mo__n_estimators':[100,200],'mo__min_samples_split': [2,3,5,10]},
    {'mo__min_samples_leaf' : [3,5,7,10],'mo__n_jobs' : [-1,2,3]}
]


# Parameters = [
#     {'randomforestregressor__n_estimators':[100,200],'randomforestregressor__max_depth' : [10,12],'randomforestregressor__n_jobs' : [-1]},
#     {'randomforestregressor__n_estimators':[100,200],'randomforestregressor__max_depth' : [6,8,10,12]},
#     {'randomforestregressor__min_samples_leaf' : [3,5,7,10],'randomforestregressor__min_samples_split': [2,3,5,10]},
#     {'randomforestregressor__n_estimators':[100,200],'randomforestregressor__min_samples_split': [2,3,5,10]},
#     {'randomforestregressor__min_samples_leaf' : [3,5,7,10],'randomforestregressor__n_jobs' : [-1,2,3]}
# ]

pipe = Pipeline([('scaler',StandardScaler()),('mo',RandomForestRegressor())]) # (pipe) 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다
# pipe = make_pipeline(MinMaxScaler(),RandomForestRegressor())

models = [GridSearchCV(pipe,Parameters,cv = 5),RandomizedSearchCV(pipe,Parameters,cv = 5)]

for algorithm in models :  
    model = algorithm
    model.fit(x_train, y_train)
    print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 
    print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다

# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=7, n_jobs=2)
# 최종정답률 :  0.5368903101355602
# 최종정답률 :  0.5368903101355601

# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, min_samples_split=5)
# 최종정답률 :  0.5156058212281027
# 최종정답률 :  0.5156058212281027

# M
# 최종정답률 :  0.5107361886475568

# S
# 최종정답률 :  0.512618329482117

# 최적의 매개변수 :  Pipeline(steps=[('scaler', StandardScaler()),
#                 ('mo',
#                  RandomForestRegressor(min_samples_leaf=10,
#                                        min_samples_split=3))])
# 최종정답률 :  0.5112348674405847
# 최적의 매개변수 :  Pipeline(steps=[('scaler', StandardScaler()),
#                 ('mo',
#                  RandomForestRegressor(min_samples_leaf=7,
#                                        min_samples_split=3))])
# 최종정답률 :  0.5222610746511505