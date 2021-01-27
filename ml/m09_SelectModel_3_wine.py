from sklearn.model_selection import train_test_split
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

allAlgorithms = all_estimators(type_filter='classifier')

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        # continue
        print(name,'없는 모델')

import sklearn
print(sklearn.__version__) # 0.23.2



# AdaBoostClassifier 의 정답률 :  0.8888888888888888
# BaggingClassifier 의 정답률 :  0.8888888888888888
# BernoulliNB 의 정답률 :  0.3333333333333333
# CalibratedClassifierCV 의 정답률 :  0.9166666666666666
# CategoricalNB 없는 모델
# CheckingClassifier 의 정답률 :  0.3888888888888889
# ClassifierChain 없는 모델
# ComplementNB 의 정답률 :  0.6666666666666666
# DecisionTreeClassifier 의 정답률 :  0.8333333333333334
# DummyClassifier 의 정답률 :  0.3333333333333333
# ExtraTreeClassifier 의 정답률 :  0.8888888888888888
# ExtraTreesClassifier 의 정답률 :  0.9444444444444444
# GaussianNB 의 정답률 :  0.9444444444444444
# GaussianProcessClassifier 의 정답률 :  0.4444444444444444
# GradientBoostingClassifier 의 정답률 :  0.9166666666666666
# HistGradientBoostingClassifier 의 정답률 :  0.9444444444444444
# KNeighborsClassifier 의 정답률 :  0.7777777777777778
# LabelPropagation 의 정답률 :  0.4444444444444444
# LabelSpreading 의 정답률 :  0.4444444444444444
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.8055555555555556
# LogisticRegression 의 정답률 :  0.9444444444444444
# LogisticRegressionCV 의 정답률 :  0.9166666666666666
# MLPClassifier 의 정답률 :  0.4722222222222222
# MultiOutputClassifier 없는 모델
# MultinomialNB 의 정답률 :  0.9444444444444444
# NearestCentroid 의 정답률 :  0.8055555555555556
# NuSVC 의 정답률 :  0.8055555555555556
# OneVsOneClassifier 없는 모델
# OneVsRestClassifier 없는 모델
# OutputCodeClassifier 없는 모델
# PassiveAggressiveClassifier 의 정답률 :  0.6111111111111112
# Perceptron 의 정답률 :  0.6111111111111112
# QuadraticDiscriminantAnalysis 의 정답률 :  1.0
# RadiusNeighborsClassifier 없는 모델
# RandomForestClassifier 의 정답률 :  0.9444444444444444
# RidgeClassifier 의 정답률 :  1.0
# RidgeClassifierCV 의 정답률 :  1.0
# SGDClassifier 의 정답률 :  0.6388888888888888
# SVC 의 정답률 :  0.6666666666666666
# StackingClassifier 없는 모델
# VotingClassifier 없는 모델