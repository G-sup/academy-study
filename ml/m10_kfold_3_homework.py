# trian test 나눈다음에 train만 발리데이션 하지말고 
# kfold 한후에 train_test_split사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')


# 1 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape)        # (96, 4)
print(y.shape)        # (96, 4)

KFold = KFold(n_splits=5,shuffle=True,random_state=104) # (shuffle=False : 순차적)

x_test = []
y_test = []
x_train = []
y_train = []
x_val = []
y_val = []

for train_index, test_index in KFold.split(x,y):

    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 104)



    

print(x_train.shape)        # (96, 4)
print(x_test.shape)         # (30, 4)
print(x_val.shape)          # (24, 4)

print(y_train.shape)        # (96, )
print(y_test.shape)         # (30, )
print(y_val.shape)          # (24, )

           


#2 모델 구성

model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

scores = cross_val_score(model, x_train, y_train, cv=KFold)

print('scores : ',scores)

#3 훈련
'''
# model.fit(x_train, y_train)

#4 평가 예측


y_pred = model.predict(x_test)
# print(x_test,"'s result : ",y_pred)


result = model.score(x_test, y_test)
print('modle_socore : ',result)

acc = accuracy_score(y_test,y_pred)
print('accuracy_score : ',acc)

# =====LinearSVC=====
# modle_socore :  0.9333333333333333
# accuracy_score :  0.9333333333333333

# =====SVC=====
# modle_socore :  1.0
# accuracy_score :  1.0

# =====KNeighborsClassifier=====
# modle_socore :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# =====DecisionTreeClassifier=====
# modle_socore :  1.0
# accuracy_score :  1.0

# ======RandomForestClassifier=====
# modle_socore :  1.0
# accuracy_score :  1.0

# =====LogisticRegression=====
# modle_socore :  0.9
# accuracy_score :  0.9

# =====Tensorflow=====
# [1.0]
'''