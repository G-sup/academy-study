# Classifier = 분류 EX) model = ~~~~~~Classifier


import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC,LinearSVR,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

#2 모델 구성       

models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=KFold) # accuracy_score
    print(algorithm)
    print('scores : ', scores)

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

# scores = cross_val_score(model, x_train, y_train, cv=KFold)

# print('scores : ',scores)

# #3 훈련

# model.fit(x_train, y_train)

# #4 평가 예측


# y_pred = model.predict(x_test)
# # print(x_test,"'s result : ",y_pred)


# result = model.score(x_test, y_test)
# print('modle_socore : ',result)

# acc = accuracy_score(y_test,y_pred)
# print('accuracy_score : ',acc)


# =====LinearSVC=====
# 0.94505495
# modle_socore :  0.9736842105263158
# accuracy_score :  0.9736842105263158

# =====SVC=====
# 0.95604396
# modle_socore :  0.9824561403508771
# accuracy_score :  0.9824561403508771

# =====KNeighborsClassifier=====
# 0.95604396
# modle_socore :  0.956140350877193
# accuracy_score :  0.956140350877193

# =====DecisionTreeClassifier=====
# 0.94505495
# modle_socore :  0.9122807017543859
# accuracy_score :  0.9122807017543859

# ======RandomForestClassifier=====
# 0.95604396
# modle_socore :  0.956140350877193
# accuracy_score :  0.956140350877193

# =====LogisticRegression=====
# 0.97802198
# modle_socore :  0.9649122807017544
# accuracy_score :  0.9649122807017544

# =====Tensorflow=====
# [0.9649122953414917]
