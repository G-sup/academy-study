from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
# from sklearn.utils import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
#  [0.98901099 0.97802198 0.96703297 0.96703297 0.93406593]
# BaggingClassifier 의 정답률 : 
#  [0.97802198 0.9010989  0.95604396 0.97802198 0.96703297]
# BernoulliNB 의 정답률 :
#  [0.7032967  0.61538462 0.65934066 0.62637363 0.56043956]
# CalibratedClassifierCV 의 정답률 : 
#  [0.87912088 0.9010989  0.96703297 0.93406593 0.91208791]
# CategoricalNB 없는 모델
# CheckingClassifier 의 정답률 :
#  [0. 0. 0. 0. 0.]
# ClassifierChain 없는 모델
# ComplementNB 의 정답률 :
#  [0.89010989 0.89010989 0.89010989 0.9010989  0.92307692]
# DecisionTreeClassifier 의 정답률 : 
#  [0.89010989 0.94505495 0.9010989  0.93406593 0.94505495]
# DummyClassifier 의 정답률 :
#  [0.46153846 0.56043956 0.6043956  0.46153846 0.50549451]
# ExtraTreeClassifier 의 정답률 :
#  [0.92307692 0.89010989 0.9010989  0.91208791 0.91208791]
# ExtraTreesClassifier 의 정답률 : 
#  [0.9010989  0.95604396 0.97802198 0.98901099 0.97802198]
# GaussianNB 의 정답률 :
#  [0.97802198 0.96703297 0.89010989 0.93406593 0.93406593]
# GaussianProcessClassifier 의 정답률 : 
#  [0.87912088 0.89010989 0.89010989 0.94505495 0.93406593]
# GradientBoostingClassifier 의 정답률 : 
#  [0.98901099 0.97802198 0.94505495 0.95604396 0.95604396]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.97802198 0.98901099 0.96703297 0.96703297 0.95604396]
# KNeighborsClassifier 의 정답률 :
#  [0.92307692 0.95604396 0.91208791 0.96703297 0.91208791]
# LabelPropagation 의 정답률 : 
#  [0.36263736 0.37362637 0.32967033 0.41758242 0.43956044]
# LabelSpreading 의 정답률 : 
#  [0.31868132 0.48351648 0.35164835 0.36263736 0.36263736]
# LinearDiscriminantAnalysis 의 정답률 :
#  [0.93406593 0.96703297 0.93406593 0.93406593 0.97802198]
# LinearSVC 의 정답률 : 
#  [0.94505495 0.94505495 0.91208791 0.92307692 0.83516484]
# LogisticRegression 의 정답률 : 
#  [0.95604396 0.95604396 0.95604396 0.94505495 0.94505495]
# LogisticRegressionCV 의 정답률 : 
#  [0.97802198 0.96703297 0.94505495 0.96703297 0.92307692]
# MLPClassifier 의 정답률 : 
#  [0.93406593 0.92307692 0.91208791 0.9010989  0.95604396]
# MultiOutputClassifier 없는 모델
# MultinomialNB 의 정답률 :
#  [0.89010989 0.93406593 0.9010989  0.89010989 0.85714286]
# NearestCentroid 의 정답률 : 
#  [0.9010989  0.86813187 0.85714286 0.95604396 0.89010989]
# NuSVC 의 정답률 : 
#  [0.86813187 0.9010989  0.87912088 0.89010989 0.83516484]
# OneVsOneClassifier 없는 모델
# OneVsRestClassifier 없는 모델
# OutputCodeClassifier 없는 모델
# PassiveAggressiveClassifier 의 정답률 : 
#  [0.84615385 0.87912088 0.87912088 0.93406593 0.87912088]
# Perceptron 의 정답률 : 
#  [0.78021978 0.89010989 0.9010989  0.45054945 0.87912088]
# QuadraticDiscriminantAnalysis 의 정답률 :
#  [0.91208791 0.91208791 0.93406593 0.96703297 0.95604396]
# RadiusNeighborsClassifier 없는 모델
# RandomForestClassifier 의 정답률 : 
#  [0.97802198 0.96703297 0.94505495 0.92307692 0.95604396]
# RidgeClassifier 의 정답률 :
#  [0.97802198 0.96703297 0.91208791 0.98901099 0.89010989]
# RidgeClassifierCV 의 정답률 :
#  [0.9010989  0.94505495 1.         0.95604396 0.94505495]
# SGDClassifier 의 정답률 :
#  [0.86813187 0.89010989 0.87912088 0.76923077 0.95604396]
# SVC 의 정답률 : 
#  [0.91208791 0.91208791 0.93406593 0.9010989  0.91208791]
# StackingClassifier 없는 모델
# VotingClassifier 없는 모델
# 0.23.2