from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV # Gridserch 모델셀렉션에 있다
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 1 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

Parameters = [
    {"C" : [1,10,100,1000],"kernel":["linear"] },
    {"C" : [1,10,100],"kernel":['rbf'],"gamma":[0.001,0.0001]},
    {"C" : [1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}
]

#2 모델 구성

model = GridSearchCV(SVC(), Parameters, cv = KFold  ) 
# GridSearchCV 뒤에 모델(SVC)을  파라미터에 (감싸서) 맞춰서 돌린다 (파라미터 18 * kfold 횟수 5) 즉 총 90번이 돌아갔다.

score = cross_val_score(model, x_train,y_train,cv= KFold)

print(score)

'''
#3 훈련

model.fit(x_train, y_train)

#4 평가 예측

print('최적의 매개변수 : ', model.best_estimator_) # model.best_estimator_ : 어떤것이 가장 좋은것(매개변수)인지 나온다 

y_pred = model.predict(x_test)
print("최종정답률 : ",accuracy_score(y_test,y_pred))

print("최종정답률 : ", model.score(x_test,y_test)) # GridSearchCV 가 모델자체가 된다
'''