# Regressor = 회귀모델 EX) model = ~~~~~~Regressor 단 LogisticRegression는 분류모델


import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)



#2 모델 구성

models = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=KFold) # r2_score
    print(algorithm)
    print('scores : ', scores)  

# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# scores = cross_val_score(model, x_train, y_train, cv=KFold)

# print('scores : ',scores)

# #3 훈련

# model.fit(x_train, y_train)

# #4 평가 예측


# y_pred = model.predict(x_test)
# # print(x_test,"'s result : ",y_pred)


# result = model.score(x_test, y_test)
# print('modle_socore : ',result)

# r2 = r2_score(y_test,y_pred)
# print('r2_score : ',r2)

# =====LinearRegression=====
# modle_socore :  0.6305664839493841
# r2_score :  0.6305664839493841


# =====KNeighborsRegressor=====
# modle_socore :  0.6753314260784092
# r2_score :  0.6753314260784092


# =====DecisionTreeRegressor=====
# modle_socore :  0.8034092490220466
# r2_score :  0.8034092490220466


# =====RandomForestRegressor=====
# modle_socore :  0.8678564347474306
# r2_score :  0.8678564347474306


# =====Tensorflow=====
# R2:  0.8828222618320316