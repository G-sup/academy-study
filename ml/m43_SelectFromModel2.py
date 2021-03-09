# 1. 상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델구성 
# 최적의 R2와 피쳐인포턴스 구할것

# 2. 위 스레드 값으로 SelectFromModel을 구해서 최적의 피쳐갯수 구할것

# 3. 위 피쳐 갯수로 데이터를 수정하여 그리드서치 랜덤서치적용해서 최적의 R2 구할것

import sklearn
from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn. feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from xgboost.sklearn import XGBRegressor
warnings.filterwarnings('ignore')

# 1 데이터

x, y = load_boston(return_X_y=True)


x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1,0.3,0.001,0.01], 'max_depth':[4,5,6]},
    {'n_estimators':[90,100,110], 'learning_rate':[0.1,0.001,0.01], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators':[100,110], 'learning_rate':[0.1,0.5,0.001], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.9]}
]

#2 모델 구성

model = RandomizedSearchCV(XGBRegressor(eval_metric='mlogloss'), parameters, cv = KFold  ) 

score = cross_val_score(model, x_train,y_train,cv= KFold)

print(score)

model.fit(x_train, y_train)

print(model.best_estimator_)


thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)


# for thresh in thresholds:
#     selection = SelectFromModel(model.best_estimator_, threshold=thresh)#, prefit=True)

#     selection.fit(x_train,y_train)

#     print(x_train.shape)

#     # selection_model = XGBRegressor(n_jobs= 8)
#     selection_model  = RandomizedSearchCV(XGBRegressor(eval_metric='mlogloss'), parameters, cv = KFold  ) 

#     selection_model.fit(x_train,y_train)

#     y_predict = selection_model.predict(x_test)

#     score = r2_score(y_test,y_predict)
#     print("Thresh = %.3f , n = %d, R2 : %.2f%%" %(thresh,x_train.shape[1],score*100))


for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    # selection_model = XGBRegressor(n_jobs= 8)
    selection_model  = RandomizedSearchCV(XGBRegressor(eval_metric='mlogloss'), parameters, cv = KFold  ) 

    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)
    print("Thresh = %.3f , n = %d, R2 : %.2f%%" %(thresh,select_x_train.shape[1],score*100))



#feature important 를 쓸수있는 모델이나 랜덤포래스트등에서 사용한다

# =============================================================================
# GridSearchCV

# [0.88092816 0.84611536 0.90905591 0.84808278 0.65291045]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.3, max_delta_step=0,
#              max_depth=4, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=200, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.00943733 0.01035738 0.0103998  0.01069269 0.01325761 0.01404423
#  0.03111622 0.03382964 0.04309482 0.04691511 0.09230509 0.2122821
#  0.47226796]
# (404, 13)
# Thresh = 0.009 , n = 13, R2 : 85.41%
# (404, 12)
# Thresh = 0.010 , n = 12, R2 : 84.77%
# (404, 11)
# Thresh = 0.010 , n = 11, R2 : 85.53%
# (404, 10)
# Thresh = 0.011 , n = 10, R2 : 86.84%
# (404, 9)
# Thresh = 0.013 , n = 9, R2 : 87.17%
# (404, 8)
# Thresh = 0.014 , n = 8, R2 : 82.88%
# (404, 7)
# Thresh = 0.031 , n = 7, R2 : 83.75%
# (404, 6)
# Thresh = 0.034 , n = 6, R2 : 83.18%
# (404, 5)
# Thresh = 0.043 , n = 5, R2 : 84.79%
# (404, 4)
# Thresh = 0.047 , n = 4, R2 : 85.59%
# (404, 3)
# Thresh = 0.092 , n = 3, R2 : 80.02%
# (404, 2)
# Thresh = 0.212 , n = 2, R2 : 65.13%
# (404, 1)
# Thresh = 0.472 , n = 1, R2 : 49.95%

# =============================================================================
# RandomizedSearchCV

# [0.877282   0.87860232 0.8501936  0.82163563 0.90370981]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#              max_depth=5, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=110, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.00261595 0.00776907 0.0084966  0.00955206 0.01321212 0.01644552
#  0.02217321 0.02352873 0.0368558  0.04691745 0.05074593 0.27868906
#  0.48299858]
# (404, 13)
# Thresh = 0.003 , n = 13, R2 : 85.41%
# (404, 12)
# Thresh = 0.008 , n = 12, R2 : 85.88%
# (404, 11)
# Thresh = 0.008 , n = 11, R2 : 84.80%
# (404, 10)
# Thresh = 0.010 , n = 10, R2 : 84.98%
# (404, 9)
# Thresh = 0.013 , n = 9, R2 : 84.77%
# (404, 8)
# Thresh = 0.016 , n = 8, R2 : 87.20%
# (404, 7)
# Thresh = 0.022 , n = 7, R2 : 83.75%
# (404, 6)
# Thresh = 0.024 , n = 6, R2 : 86.23%
# (404, 5)
# Thresh = 0.037 , n = 5, R2 : 82.99%
# (404, 4)
# Thresh = 0.047 , n = 4, R2 : 83.42%
# (404, 3)
# Thresh = 0.051 , n = 3, R2 : 80.80%
# (404, 2)
# Thresh = 0.279 , n = 2, R2 : 65.13%
# (404, 1)
# Thresh = 0.483 , n = 1, R2 : 49.95%


# ============================================================================

# [0.86315932 0.83319392 0.89670871 0.86472813 0.89209175]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#              max_depth=6, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=110, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.00309774 0.00786463 0.00881752 0.00920683 0.00966451 0.01446413
#  0.02400722 0.03031676 0.03346439 0.04635575 0.04897222 0.23988178
#  0.52388656]
# (404, 13)
# Thresh = 0.003 , n = 13, R2 : 87.82%
# (404, 12)
# Thresh = 0.008 , n = 12, R2 : 91.42%
# (404, 11)
# Thresh = 0.009 , n = 11, R2 : 88.93%
# (404, 10)
# Thresh = 0.009 , n = 10, R2 : 90.06%
# (404, 9)
# Thresh = 0.010 , n = 9, R2 : 89.44%
# (404, 8)
# Thresh = 0.014 , n = 8, R2 : 86.93%
# (404, 7)
# Thresh = 0.024 , n = 7, R2 : 86.60%
# (404, 6)
# Thresh = 0.030 , n = 6, R2 : 87.67%
# (404, 5)
# Thresh = 0.033 , n = 5, R2 : 86.16%
# (404, 4)
# Thresh = 0.046 , n = 4, R2 : 88.59%
# (404, 3)
# Thresh = 0.049 , n = 3, R2 : 86.59%
# (404, 2)
# Thresh = 0.240 , n = 2, R2 : 66.22%
# (404, 1)
# Thresh = 0.524 , n = 1, R2 : 60.72%
