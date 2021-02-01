# randomforest
# pipeline

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

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

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

model = RandomizedSearchCV(pipe,Parameters,cv = KFold) 

# score = cross_val_score(model, x_train,y_train,cv= KFold)
score = cross_val_score(model, x, y, cv= KFold,verbose=1)

print(score)


