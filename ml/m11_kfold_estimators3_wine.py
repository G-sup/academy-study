from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
# from sklearn.utils import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)
KFold = KFold(n_splits=5,shuffle=True) # (n_splits=5 : 몇개로 나눌것인가 ,shuffle=False : 순차적)


allAlgorithms = all_estimators(type_filter='classifier')

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=KFold) # kfold 대신에 숫자를 넣어도 된다 단 셔플X


        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name,'의 정답률 : \n', scores)
    except:
        # continue
        print(name,'없는 모델')

import sklearn
print(sklearn.__version__) # 0.23.2

# AdaBoostClassifier 의 정답률 : 
#  [0.89655172 0.89655172 1.         0.92857143 1.        ]
# BaggingClassifier 의 정답률 : 
#  [0.96551724 0.96551724 1.         1.         0.96428571]
# BernoulliNB 의 정답률 :
#  [0.51724138 0.4137931  0.42857143 0.39285714 0.32142857]
# CalibratedClassifierCV 의 정답률 : 
#  [0.93103448 0.93103448 0.96428571 0.85714286 0.89285714]
# CategoricalNB 없는 모델
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 없는 모델
# ComplementNB 의 정답률 :
#  [0.5862069  0.5862069  0.60714286 0.82142857 0.57142857]
# DecisionTreeClassifier 의 정답률 : 
#  [0.86206897 0.89655172 0.92857143 0.96428571 0.92857143]
# DummyClassifier 의 정답률 :
#  [0.48275862 0.34482759 0.39285714 0.25       0.39285714]
# ExtraTreeClassifier 의 정답률 :
#  [0.79310345 0.82758621 0.96428571 0.78571429 0.89285714]
# ExtraTreesClassifier 의 정답률 : 
#  [1. 1. 1. 1. 1.]
# GaussianNB 의 정답률 :
#  [1.         1.         1.         0.96428571 1.        ]
# GaussianProcessClassifier 의 정답률 : 
#  [0.48275862 0.48275862 0.46428571 0.53571429 0.35714286]
# GradientBoostingClassifier 의 정답률 : 
#  [0.96551724 0.93103448 0.89285714 0.92857143 1.        ]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.96551724 1.         0.96428571 1.         1.        ]
# KNeighborsClassifier 의 정답률 :
#  [0.62068966 0.55172414 0.60714286 0.71428571 0.67857143]
# LabelPropagation 의 정답률 :
#  [0.51724138 0.4137931  0.28571429 0.5        0.64285714]
# LabelSpreading 의 정답률 : 
#  [0.65517241 0.4137931  0.35714286 0.39285714 0.67857143]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [0.96551724 1.         0.96428571 1.         0.96428571]
# LinearSVC 의 정답률 : 
#  [0.82758621 0.79310345 0.89285714 0.71428571 0.89285714]
# LogisticRegression 의 정답률 : 
#  [0.96551724 0.93103448 1.         0.96428571 0.96428571]
# LogisticRegressionCV 의 정답률 : 
#  [1.         0.93103448 0.85714286 1.         0.89285714]
# MLPClassifier 의 정답률 : 
#  [0.75862069 0.65517241 0.35714286 0.39285714 0.42857143]
# MultiOutputClassifier 없는 모델
# MultinomialNB 의 정답률 :
#  [0.93103448 0.75862069 0.78571429 0.92857143 0.82142857]
# NearestCentroid 의 정답률 :
#  [0.65517241 0.79310345 0.64285714 0.71428571 0.67857143]
# NuSVC 의 정답률 : 
#  [0.79310345 0.96551724 0.92857143 0.89285714 0.85714286]
# OneVsOneClassifier 없는 모델
# OneVsRestClassifier 없는 모델
# OutputCodeClassifier 없는 모델
# PassiveAggressiveClassifier 의 정답률 : 
#  [0.48275862 0.51724138 0.46428571 0.75       0.57142857]
# Perceptron 의 정답률 :
#  [0.75862069 0.72413793 0.32142857 0.5        0.60714286]
# QuadraticDiscriminantAnalysis 의 정답률 : 
#  [1.         0.96551724 1.         1.         1.        ]
# RadiusNeighborsClassifier 없는 모델
# RandomForestClassifier 의 정답률 : 
#  [1. 1. 1. 1. 1.]
# RidgeClassifier 의 정답률 :
#  [1.         0.96551724 1.         0.96428571 1.        ]
# RidgeClassifierCV 의 정답률 : 
#  [0.96551724 1.         1.         1.         0.96428571]
# SGDClassifier 의 정답률 :
#  [0.48275862 0.51724138 0.5        0.53571429 0.46428571]
# SVC 의 정답률 : 
#  [0.62068966 0.62068966 0.78571429 0.71428571 0.67857143]
# StackingClassifier 없는 모델
# VotingClassifier 없는 모델
# 0.23.2