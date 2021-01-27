from sklearn.model_selection import train_test_split
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



# AdaBoostClassifier 의 정답률 :  0.9736842105263158
# BaggingClassifier 의 정답률 :  0.9473684210526315
# BernoulliNB 의 정답률 :  0.6052631578947368
# CalibratedClassifierCV 의 정답률 :  0.9298245614035088
# CategoricalNB 없는 모델
# CheckingClassifier 의 정답률 :  0.39473684210526316
# ClassifierChain 없는 모델
# ComplementNB 의 정답률 :  0.8947368421052632
# DecisionTreeClassifier 의 정답률 :  0.9298245614035088
# DummyClassifier 의 정답률 :  0.5263157894736842
# ExtraTreeClassifier 의 정답률 :  0.9035087719298246
# ExtraTreesClassifier 의 정답률 :  0.956140350877193
# GaussianNB 의 정답률 :  0.9210526315789473
# GaussianProcessClassifier 의 정답률 :  0.9298245614035088
# GradientBoostingClassifier 의 정답률 :  0.9385964912280702
# HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
# KNeighborsClassifier 의 정답률 :  0.9473684210526315
# LabelPropagation 의 정답률 :  0.42105263157894735
# LabelSpreading 의 정답률 :  0.42105263157894735
# LinearDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# LinearSVC 의 정답률 :  0.8859649122807017
# LogisticRegression 의 정답률 :  0.9298245614035088
# LogisticRegressionCV 의 정답률 :  0.956140350877193
# MLPClassifier 의 정답률 :  0.9210526315789473
# MultiOutputClassifier 없는 모델
# MultinomialNB 의 정답률 :  0.8947368421052632
# NearestCentroid 의 정답률 :  0.868421052631579
# NuSVC 의 정답률 :  0.8508771929824561
# OneVsOneClassifier 없는 모델
# OneVsRestClassifier 없는 모델
# OutputCodeClassifier 없는 모델
# PassiveAggressiveClassifier 의 정답률 :  0.9298245614035088
# Perceptron 의 정답률 :  0.9385964912280702
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9736842105263158
# RadiusNeighborsClassifier 없는 모델
# RandomForestClassifier 의 정답률 :  0.956140350877193
# RidgeClassifier 의 정답률 :  0.9736842105263158
# RidgeClassifierCV 의 정답률 :  0.9649122807017544
# SGDClassifier 의 정답률 :  0.8333333333333334
# SVC 의 정답률 :  0.9210526315789473
# StackingClassifier 없는 모델
# VotingClassifier 없는 모델