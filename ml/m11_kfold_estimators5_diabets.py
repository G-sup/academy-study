from sklearn.model_selection import train_test_split,KFold,cross_val_score
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
KFold = KFold(n_splits=5,shuffle=True) # (n_splits=5 : 몇개로 나눌것인가 ,shuffle=False : 순차적)

allAlgorithms = all_estimators(type_filter='regressor')

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


# ARDRegression 의 정답률 : 
#  [0.47727076 0.42878311 0.42297787 0.4768541  0.58443045]
# AdaBoostRegressor 의 정답률 : 
#  [0.49928681 0.28545925 0.5029506  0.49513016 0.34492884]
# BaggingRegressor 의 정답률 : 
#  [0.43675168 0.54860415 0.33762464 0.09652464 0.32567389]
# BayesianRidge 의 정답률 :
#  [0.33959871 0.43316375 0.51668149 0.5558023  0.5574687 ]
# CCA 의 정답률 : 
#  [0.34685674 0.48577109 0.34866194 0.38399651 0.4793815 ]
# DecisionTreeRegressor 의 정답률 : 
#  [-0.16186258 -0.15802773 -0.323369    0.09913472  0.08055254]
# DummyRegressor 의 정답률 :
#  [-0.00038692 -0.00536168 -0.0006196  -0.00012478 -0.00209976]
# ElasticNet 의 정답률 :
#  [-0.10175236  0.00481433  0.00243721 -0.02156436  0.00763551]
# ElasticNetCV 의 정답률 : 
#  [0.41587839 0.49216296 0.41453966 0.40059267 0.38436645]
# ExtraTreeRegressor 의 정답률 :
#  [-0.43554827 -0.19435979 -0.06436111  0.10103988  0.11160221]
# ExtraTreesRegressor 의 정답률 : 
#  [0.39310039 0.4799858  0.51793183 0.39912245 0.42664426]
# GammaRegressor 의 정답률 :
#  [ 0.00242133  0.00043372 -0.00889821  0.00396484  0.00480573]
# GaussianProcessRegressor 의 정답률 : 
#  [-16.54921542 -17.6324844  -14.1196406  -28.58559385  -8.08507648]
# GeneralizedLinearRegressor 의 정답률 :
#  [-0.03251757 -0.00123612  0.00092581 -0.02922763  0.00598659]
# GradientBoostingRegressor 의 정답률 : 
#  [0.3270046  0.23067644 0.46759063 0.48083482 0.28823526]
# HistGradientBoostingRegressor 의 정답률 : 
#  [0.07916258 0.27540977 0.51808019 0.38729271 0.25644854]
# HuberRegressor 의 정답률 : 
#  [0.33504239 0.5597709  0.59435817 0.44034214 0.29482106]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 : 
#  [0.19315272 0.3730534  0.26501044 0.42433454 0.35014191]
# KernelRidge 의 정답률 :
#  [-4.18882031 -2.88881808 -3.69645623 -3.10500356 -4.46243548]
# Lars 의 정답률 : 
#  [ 0.28885719  0.38232818  0.4674317  -1.44330915  0.51288196]
# LarsCV 의 정답률 : 
#  [0.46284311 0.40668674 0.55506965 0.49653882 0.31675618]
# Lasso 의 정답률 :
#  [0.3022695  0.36284612 0.38755813 0.2795157  0.34427891]
# LassoCV 의 정답률 : 
#  [0.41695958 0.49675139 0.58470972 0.3145448  0.48450424]
# LassoLars 의 정답률 :
#  [0.35582856 0.42897526 0.35507438 0.33515597 0.359124  ]
# LassoLarsCV 의 정답률 : 
#  [0.4298724  0.27017454 0.56138777 0.59627653 0.32398603]
# LassoLarsIC 의 정답률 : 
#  [0.28779441 0.3750803  0.5191614  0.55948703 0.52395435]
# LinearRegression 의 정답률 :
#  [0.55315594 0.47106523 0.4202685  0.47753857 0.34372771]
# LinearSVR 의 정답률 :
#  [-0.34164442 -0.33497105 -0.84724209 -0.60251158 -0.29355448]
# MLPRegressor 의 정답률 : 
#  [-2.60003425 -3.10474465 -3.31648055 -3.11025254 -3.2992928 ]
# MultiOutputRegressor 없는 모델
# MultiTaskElasticNet 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 :
#  [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 : 
#  [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :
#  [nan nan nan nan nan]
# NuSVR 의 정답률 : 
#  [0.13391959 0.11221407 0.0996949  0.11733068 0.12292389]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.26567454 0.35608703 0.43972105 0.33538958 0.30353385]
# OrthogonalMatchingPursuitCV 의 정답률 : 
#  [0.3699216  0.30638755 0.41174308 0.4830534  0.60826998]
# PLSCanonical 의 정답률 :
#  [-0.37598605 -1.63747771 -1.25943616 -1.10194781 -1.51213047]
# PLSRegression 의 정답률 : 
#  [0.45130536 0.313079   0.55599636 0.5129716  0.52987851]
# PassiveAggressiveRegressor 의 정답률 :
#  [0.46830713 0.45160096 0.31710779 0.45685711 0.39226348]
# PoissonRegressor 의 정답률 : 
#  [0.26889651 0.27411482 0.20060346 0.3532167  0.32169555]
# RANSACRegressor 의 정답률 : 
#  [-0.23629138 -0.25114558 -0.3100847   0.39947635  0.23111175]
# RadiusNeighborsRegressor 의 정답률 : 
#  [-2.26083114e-02 -2.56249506e-02 -1.31197289e-07 -6.32438074e-02
#  -9.26273941e-02]
# RandomForestRegressor 의 정답률 : 
#  [0.30157312 0.32468314 0.51694515 0.40092454 0.3518868 ]
# RegressorChain 없는 모델
# Ridge 의 정답률 :
#  [0.3327903  0.4192151  0.38766378 0.40160418 0.39990883]
# RidgeCV 의 정답률 :
#  [0.41937975 0.50244852 0.48276185 0.53034146 0.42554357]
# SGDRegressor 의 정답률 : 
#  [0.36347648 0.37010394 0.44630646 0.2766891  0.39295172]
# SVR 의 정답률 :
#  [0.05318991 0.1243684  0.13200921 0.08562996 0.14076383]
# StackingRegressor 없는 모델
# TheilSenRegressor 의 정답률 : 
#  [0.38934179 0.17641735 0.5480753  0.56211004 0.52832268]
# TransformedTargetRegressor 의 정답률 :
#  [0.43161819 0.5215957  0.58537291 0.49870494 0.32643748]
# TweedieRegressor 의 정답률 : 
#  [-0.00254066 -0.00237796  0.00312014  0.0059198  -0.00615498]
# VotingRegressor 없는 모델
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]
# 0.23.2