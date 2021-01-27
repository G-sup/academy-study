from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
# from sklearn.utils import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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



# ARDRegression 의 정답률 :  0.5500097935622059
# AdaBoostRegressor 의 정답률 :  0.46818886018080874
# BaggingRegressor 의 정답률 :  0.553844255327256
# BayesianRidge 의 정답률 :  0.5586321601529097
# CCA 의 정답률 :  0.5513342929728616
# DecisionTreeRegressor 의 정답률 :  -0.04717392290283828
# DummyRegressor 의 정답률 :  -0.00042280247579373764
# ElasticNet 의 정답률 :  0.008976297954434886
# ElasticNetCV 의 정답률 :  0.495696121136317
# ExtraTreeRegressor 의 정답률 :  -0.02420645638696639
# ExtraTreesRegressor 의 정답률 :  0.5156167522115517
# GammaRegressor 의 정답률 :  0.006740425970623964
# GaussianProcessRegressor 의 정답률 :  -10.08340956226191
# GeneralizedLinearRegressor 의 정답률 :  0.0068033400003642
# GradientBoostingRegressor 의 정답률 :  0.5153681815476895
# HistGradientBoostingRegressor 의 정답률 :  0.4626189626435241
# HuberRegressor 의 정답률 :  0.5612152734618563
# IsotonicRegression 없는 모델
# KNeighborsRegressor 의 정답률 :  0.4845168685552085
# KernelRidge 의 정답률 :  -3.3933664403469646
# Lars 의 정답률 :  0.5620438671817143
# LarsCV 의 정답률 :  0.5366076390799848
# Lasso 의 정답률 :  0.3524146855616862
# LassoCV 의 정답률 :  0.5518380282870776
# LassoLars 의 정답률 :  0.3826142954798889
# LassoLarsCV 의 정답률 :  0.5489375743307298
# LassoLarsIC 의 정답률 :  0.530853731221665
# LinearRegression 의 정답률 :  0.5620438671817143
# LinearSVR 의 정답률 :  -0.38278289045437197
# MLPRegressor 의 정답률 :  -3.176735487761011
# MultiOutputRegressor 없는 모델
# MultiTaskElasticNet 없는 모델
# MultiTaskElasticNetCV 없는 모델
# MultiTaskLasso 없는 모델
# MultiTaskLassoCV 없는 모델
# NuSVR 의 정답률 :  0.19061664497691422
# OrthogonalMatchingPursuit 의 정답률 :  0.28512861235694453
# OrthogonalMatchingPursuitCV 의 정답률 :  0.5484152736437475
# PLSCanonical 의 정답률 :  -1.5359148198265156
# PLSRegression 의 정답률 :  0.5627719663722044
# PassiveAggressiveRegressor 의 정답률 :  0.5534300846611054
# PoissonRegressor 의 정답률 :  0.39306895401064457
# RANSACRegressor 의 정답률 :  0.15702336586321686
# RadiusNeighborsRegressor 의 정답률 :  -0.00042280247579373764
# RandomForestRegressor 의 정답률 :  0.5036163047884735
# RegressorChain 없는 모델
# Ridge 의 정답률 :  0.47702325733640116
# RidgeCV 의 정답률 :  0.5551593176413243
# SGDRegressor 의 정답률 :  0.46704418551204496
# SVR 의 정답률 :  0.1913043362865059
# StackingRegressor 없는 모델
# TheilSenRegressor 의 정답률 :  0.5531515240072981
# TransformedTargetRegressor 의 정답률 :  0.5620438671817143
# TweedieRegressor 의 정답률 :  0.0068033400003642
# VotingRegressor 없는 모델
# _SigmoidCalibration 없는 모델