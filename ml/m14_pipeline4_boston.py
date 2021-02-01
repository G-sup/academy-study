# Regressor = 회귀모델 EX) model = ~~~~~~Regressor 단 LogisticRegression는 분류모델


import numpy as np
from sklearn.datasets import load_boston
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

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

#2 모델 구성

model = Pipeline([('scaler',StandardScaler()),('model',RandomForestRegressor())]) # (pipe) 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다
# model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

#3 훈련
model.fit(x_train, y_train)

#4 평가 예측

results = model.score(x_test,y_test)

print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_estimators=200)
# 최종정답률 :  0.8868969116318977
# 최종정답률 :  0.8868969116318977

# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, n_estimators=200)
# 최종정답률 :  0.8789847623449482
# 최종정답률 :  0.8789847623449482

# M
# 최종정답률 :  0.8807540470380921

# S
# 최종정답률 :  0.8731874142426881