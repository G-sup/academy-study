# Regressor = 회귀모델 EX) model = ~~~~~~Regressor 단 LogisticRegression는 분류모델


import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score

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

start_time = datetime.datetime.now()


KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

Parameters = [
    {'n_estimators':[100,200],'max_depth' : [10,12],'n_jobs' : [-1]},
    {'n_estimators':[100,200],'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split': [2,3,5,10]},
    {'n_estimators':[100,200],'min_samples_split': [2,3,5,10]},
    {'min_samples_leaf' : [3,5,7,10],'n_jobs' : [-1,2,3]}
]


#2 모델 구성

# model = SVC()

model = RandomizedSearchCV(RandomForestRegressor(), Parameters, cv = KFold  ) 
# RandomizedSearchCV 뒤에 모델(SVC)을  파라미터에 (감싸서) 맞춰서 돌린다 (파라미터 18 * kfold 횟수 5) 즉 총 90번이 돌아갔다.


#3 훈련

model.fit(x_train, y_train)

#4 평가 예측

print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 

y_pred = model.predict(x_test)
print("최종정답률 : ", r2_score(y_test,y_pred))

print("최종정답률 : ", model.score(x_test,y_test)) # RandomizedSearchCV 가 모델자체가 된다


end_time = datetime.datetime.now()
print('걸린시간 : ', end_time - start_time)

# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=7, n_jobs=2)
# 최종정답률 :  0.5368903101355602
# 최종정답률 :  0.5368903101355601

# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, min_samples_split=5)
# 최종정답률 :  0.5156058212281027
# 최종정답률 :  0.5156058212281027