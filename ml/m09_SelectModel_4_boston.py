from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
# from sklearn.utils import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)

allAlgorithms = all_estimators(type_filter='regressor')

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', r2_score(y_test,y_pred))
    except:
        # continue
        print(name,'없는 모델')

import sklearn
print(sklearn.__version__) # 0.23.2



# ARDRegression 의 정답률 :  0.619221538835631
# AdaBoostRegressor 의 정답률 :  0.8362485308811083
# BaggingRegressor 의 정답률 :  0.8497955470813304
# BayesianRidge 의 정답률 :  0.6200490371430183
# CCA 의 정답률 :  0.5191376723339463
# DecisionTreeRegressor 의 정답률 :  0.8246168407602343
# DummyRegressor 의 정답률 :  -0.00045472590128947665
# ElasticNet 의 정답률 :  0.621276920999314
# ElasticNetCV 의 정답률 :  0.6030072608017318
# ExtraTreeRegressor 의 정답률 :  0.6124421807568763
# ExtraTreesRegressor 의 정답률 :  0.8528257496133806
# GammaRegressor 의 정답률 :  -0.00045472590128947665
# GaussianProcessRegressor 의 정답률 :  -6.051504658774566
# GeneralizedLinearRegressor 의 정답률 :  0.6098028754100702
# GradientBoostingRegressor 의 정답률 :  0.9078364186178354
# HistGradientBoostingRegressor 의 정답률 :  0.8813318333569516
# HuberRegressor 의 정답률 :  0.43431681888645524
# IsotonicRegression 없는 모델
# KNeighborsRegressor 의 정답률 :  0.5109428046861038
# KernelRidge 의 정답률 :  0.5576758510909041
# Lars 의 정답률 :  0.6305664839493835
# LarsCV 의 정답률 :  0.626532433385923
# Lasso 의 정답률 :  0.6164107424327716
# LassoCV 의 정답률 :  0.6243322282299564
# LassoLars 의 정답률 :  -0.00045472590128947665
# LassoLarsCV 의 정답률 :  0.6290537438421882
# LassoLarsIC 의 정답률 :  0.6305664839493835
# LinearRegression 의 정답률 :  0.630566483949382
# LinearSVR 의 정답률 :  0.3349718336127866
# MLPRegressor 의 정답률 :  0.3171162610645565
# MultiOutputRegressor 없는 모델
# MultiTaskElasticNet 없는 모델
# MultiTaskElasticNetCV 없는 모델
# MultiTaskLasso 없는 모델
# MultiTaskLassoCV 없는 모델
# NuSVR 의 정답률 :  0.09804722877318062
# OrthogonalMatchingPursuit 의 정답률 :  0.507550587319902
# OrthogonalMatchingPursuitCV 의 정답률 :  0.5733871915130783
# PLSCanonical 의 정답률 :  -2.6960259609092327
# PLSRegression 의 정답률 :  0.5575917442586772
# PassiveAggressiveRegressor 의 정답률 :  0.03930267614629601
# PoissonRegressor 의 정답률 :  0.722569278467154
# RANSACRegressor 의 정답률 :  0.14365601502229985
# RadiusNeighborsRegressor 없는 모델
# RandomForestRegressor 의 정답률 :  0.8848712193833382
# RegressorChain 없는 모델
# Ridge 의 정답률 :  0.622926512186271
# RidgeCV 의 정답률 :  0.6294470208491281
# SGDRegressor 의 정답률 :  -2.2323021903868945e+26
# SVR 의 정답률 :  0.04963622795869216
# StackingRegressor 없는 모델
# TheilSenRegressor 의 정답률 :  0.483712977295668
# TransformedTargetRegressor 의 정답률 :  0.630566483949382
# TweedieRegressor 의 정답률 :  0.6098028754100702
# VotingRegressor 없는 모델
# _SigmoidCalibration 없는 모델