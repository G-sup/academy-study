# Classifier = 분류 EX) model = ~~~~~~Classifier


import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import datetime

# 1 데이터

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)


#2 모델 구성

model = Pipeline([('scaler',StandardScaler()),('model',RandomForestClassifier())]) # (pipe) 아레와 결과치는 동일 하다 이방법은 이름을 정해줄수있다
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())

#3 훈련
model.fit(x_train, y_train)

#4 평가 예측

results = model.score(x_test,y_test)

print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다

# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, n_estimators=200, n_jobs=-1)
# 최종정답률 :  0.9444444444444444
# 최종정답률 :  0.9444444444444444

# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=7, n_jobs=2)
# 최종정답률 :  0.9166666666666666
# 최종정답률 :  0.9166666666666666

# M
# 최종정답률 :  0.9444444444444444

# S
# 최종정답률 :  0.9444444444444444