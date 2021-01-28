from sklearn.model_selection import train_test_split,KFold,cross_val_score
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
#  [0.84969746 0.72978637 0.65019913 0.76544199 0.62298515]
# AdaBoostRegressor 의 정답률 : 
#  [0.71695151 0.67675709 0.80794792 0.86067748 0.88651435]
# BaggingRegressor 의 정답률 : 
#  [0.82448962 0.80691582 0.83423711 0.84689256 0.85647915]
# BayesianRidge 의 정답률 :
#  [0.80614477 0.78538307 0.7008593  0.65823144 0.71024179]
# CCA 의 정답률 :
#  [0.40003549 0.77536369 0.41155274 0.7445385  0.66164948]
# DecisionTreeRegressor 의 정답률 :
#  [0.84081837 0.71819411 0.67649655 0.79660729 0.81896669]
# DummyRegressor 의 정답률 :
#  [-0.0068099  -0.00494224 -0.00446586 -0.0032262  -0.00074044]
# ElasticNet 의 정답률 : 
#  [0.64643948 0.57557611 0.71158933 0.70120618 0.73391042]
# ElasticNetCV 의 정답률 : 
#  [0.63850352 0.61299124 0.68679804 0.72518498 0.62523687]
# ExtraTreeRegressor 의 정답률 :
#  [0.79694969 0.70556497 0.58970307 0.63430185 0.66896266]
# ExtraTreesRegressor 의 정답률 : 
#  [0.84039571 0.84963405 0.9369482  0.90039004 0.76970448]
# GammaRegressor 의 정답률 :
#  [-0.06587442 -0.01869023 -0.00447374 -0.02829758 -0.05213675]
# GaussianProcessRegressor 의 정답률 : 
#  [-4.89505046 -5.64231003 -5.72078821 -7.2161816  -6.36484418]
# GeneralizedLinearRegressor 의 정답률 : 
#  [0.77079029 0.74622816 0.71157521 0.44716254 0.69123929]
# GradientBoostingRegressor 의 정답률 : 
#  [0.78713142 0.87744069 0.95028437 0.7937135  0.93000594]
# HistGradientBoostingRegressor 의 정답률 : 
#  [0.8774852  0.90378644 0.77635398 0.74007264 0.89187551]
# HuberRegressor 의 정답률 : 
#  [0.63020874 0.59479363 0.58695164 0.59389226 0.67632117]
# IsotonicRegression 의 정답률 :
#  [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 : 
#  [0.41776597 0.48607982 0.52789387 0.50562725 0.61569385]
# KernelRidge 의 정답률 : 
#  [0.85553531 0.76431238 0.75180658 0.73138717 0.50108424]
# Lars 의 정답률 :
#  [0.67484954 0.77385904 0.63384472 0.80844794 0.80459862]
# LarsCV 의 정답률 : 
#  [0.64474586 0.64229364 0.76020541 0.77339069 0.77897693]
# Lasso 의 정답률 :
#  [0.64418832 0.70517141 0.70045631 0.73508402 0.62931521]
# LassoCV 의 정답률 : 
#  [0.69524144 0.6827722  0.66100929 0.76755535 0.6608757 ]
# LassoLars 의 정답률 :
#  [-0.00516057 -0.00348894 -0.02000651 -0.00422921 -0.01000783]
# LassoLarsCV 의 정답률 : 
#  [0.74267445 0.73612806 0.59738388 0.64791875 0.80875678]
# LassoLarsIC 의 정답률 :
#  [0.7072542  0.80250151 0.68881881 0.61162591 0.73568659]
# LinearRegression 의 정답률 :
#  [0.70828543 0.75012451 0.83891183 0.74374148 0.58065735]
# LinearSVR 의 정답률 : 
#  [ 0.3356777   0.67555614  0.65259029 -2.30293949 -0.12093947]
# MLPRegressor 의 정답률 : 
#  [0.55342315 0.28286659 0.66457245 0.57545681 0.76306628]
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
#  [0.30313596 0.24934756 0.27398677 0.13608649 0.37507497]
# OrthogonalMatchingPursuit 의 정답률 :
#  [0.55212779 0.57026144 0.11883429 0.5412448  0.53314937]
# OrthogonalMatchingPursuitCV 의 정답률 : 
#  [0.79502672 0.64761719 0.72905688 0.63105255 0.67034563]
# PLSCanonical 의 정답률 : 
#  [-2.504623   -1.87459986 -1.38961635 -2.44861068 -2.98165656]
# PLSRegression 의 정답률 :
#  [0.8410676  0.62198755 0.62365759 0.6274098  0.77800954]
# PassiveAggressiveRegressor 의 정답률 : 
#  [-0.59579758 -1.44942314 -4.10548773 -0.07423692 -0.03514821]
# PoissonRegressor 의 정답률 : 
#  [0.74136418 0.80472439 0.79289425 0.75928254 0.74063105]
# RANSACRegressor 의 정답률 : 
#  [0.58552142 0.33593882 0.6242123  0.2403876  0.66628264]
# RadiusNeighborsRegressor 없는 모델
# RandomForestRegressor 의 정답률 : 
#  [0.8632586  0.86830134 0.8941025  0.88349273 0.7475271 ]
# RegressorChain 없는 모델
# Ridge 의 정답률 :
#  [0.69974857 0.67250725 0.8460662  0.60088959 0.77672037]
# RidgeCV 의 정답률 :
#  [0.57716157 0.77438609 0.67785797 0.85715737 0.71075943]
# SGDRegressor 의 정답률 : 
#  [-7.12241489e+26 -1.06563558e+26 -5.02169634e+26 -6.19549405e+26
#  -8.69714529e+25]
# SVR 의 정답률 : 
#  [0.18450497 0.35032271 0.20558174 0.24932601 0.20025083]
# StackingRegressor 없는 모델
# TheilSenRegressor 의 정답률 : 
#  [0.74937423 0.80030109 0.78075101 0.62542172 0.61748839]
# TransformedTargetRegressor 의 정답률 :
#  [0.7368257  0.76360813 0.71045174 0.71838683 0.67917932]
# TweedieRegressor 의 정답률 : 
#  [0.71401356 0.70536098 0.59455026 0.69019734 0.56622844]
# VotingRegressor 없는 모델
# _SigmoidCalibration 의 정답률 :
#  [nan nan nan nan nan]
# 0.23.2