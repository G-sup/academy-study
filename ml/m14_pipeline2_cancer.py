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


#2
model = Pipeline([('scaler',StandardScaler()),('model',RandomForestClassifier())]) # (pipe) 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())


model.fit(x_train, y_train)

#4 평가 예측

results = model.score(x_test,y_test)

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