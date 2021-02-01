# Classifier = 분류 EX) model = ~~~~~~Classifier


import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import datetime


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

# Parameters = [
#     {'mo__n_estimators':[100,200],'mo__max_depth' : [10,12],'mo__n_jobs' : [-1]},
#     {'mo__n_estimators':[100,200],'mo__max_depth' : [6,8,10,12]},
#     {'mo__max_depth' : [6,8,10],'mo__min_samples_leaf' : [3,5,7,10]},
#     {'mo__n_estimators':[100,200],'mo__min_samples_split': [2,3,5,10]},
#     {'mo__min_samples_leaf' : [3,5,7,10],'mo__n_jobs' : [-1,2,3]}
# ]

Parameters = [
    {'randomforestclassifier__n_estimators':[100,200],'randomforestclassifier__max_depth' : [10,12],'randomforestclassifier__n_jobs' : [-1]},
    {'randomforestclassifier__n_estimators':[100,200],'randomforestclassifier__max_depth' : [6,8,10,12]},
    {'randomforestclassifier__max_depth' : [6,8,10],'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__n_estimators':[100,200],'randomforestclassifier__min_samples_split': [2,3,5,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10],'randomforestclassifier__n_jobs' : [-1,2,3]}
]

#2 모델 구성


# pipe = Pipeline([('scaler',MinMaxScaler()),('mo',RandomForestClassifier())]) 
# # 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다  이름을 정해줘야 위에 Parameters를 조정가능하다(mo__:이름으로 지정) 
pipe = make_pipeline(StandardScaler(),RandomForestClassifier()) #이걸 사용할때는 (randomforestclassifier__)로 해야 된다

models = [GridSearchCV(pipe,Parameters,cv = 5),RandomizedSearchCV(pipe,Parameters,cv = 5)]

for algorithm in models :  
    model = algorithm
    model.fit(x_train, y_train)
    print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 
    print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다




# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, n_estimators=200, n_jobs=-1)
# 최종정답률 :  0.9649122807017544
# 최종정답률 :  0.9649122807017544

# 최적의 매개변수 :  RandomForestClassifier(max_depth=8)
# 최종정답률 :  0.956140350877193
# 최종정답률 :  0.956140350877193

# M
# 최종정답률 :  0.956140350877193

# S
# 최종정답률 :  0.9649122807017544


# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mo', RandomForestClassifier(max_depth=10, n_jobs=-1))])
# 최종정답률 :  0.956140350877193

# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mo', RandomForestClassifier(max_depth=12))])
# 최종정답률 :  0.9649122807017544
