
import sklearn
from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.datasets import load_diabetes
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

x, y = load_diabetes(return_X_y=True)


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


# thresholds = np.sort(model.best_estimator_.feature_importances_)
thresholds = np.sort(model.best_estimator_.feature_importances_)

print(thresholds)


for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

    select_x_trian = selection.transform(x_train)
    print(select_x_trian.shape)

    # selection_model = XGBRegressor(n_jobs= 8)
    selection_model  = GridSearchCV(XGBRegressor(eval_metric='mlogloss'), parameters, cv = KFold  ) 

    selection_model.fit(select_x_trian,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test,y_predict)
    print("Thresh = %.3f , n = %d, R2 : %.2f%%" %(thresh,select_x_trian.shape[1],score*100))


# ================================================================================
# RandomizedSearchCV / RandomizedSearchCV

# [0.4405565  0.32355387 0.39858065 0.32677131 0.30002142]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.6, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#              max_depth=5, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=90, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.03597896 0.04519773 0.04626726 0.04748614 0.06367358 0.07373155
#  0.07851173 0.11042302 0.24021506 0.25851497]
# (353, 10)
# Thresh = 0.036 , n = 10, R2 : 48.29%
# (353, 9)
# Thresh = 0.045 , n = 9, R2 : 47.65%
# (353, 8)
# Thresh = 0.046 , n = 8, R2 : 50.75%
# (353, 7)
# Thresh = 0.047 , n = 7, R2 : 47.15%
# (353, 6)
# Thresh = 0.064 , n = 6, R2 : 53.34%
# (353, 5)
# Thresh = 0.074 , n = 5, R2 : 45.73%
# (353, 4)
# Thresh = 0.079 , n = 4, R2 : 47.18%
# (353, 3)
# Thresh = 0.110 , n = 3, R2 : 40.44%
# (353, 2)
# Thresh = 0.240 , n = 2, R2 : 38.23%
# (353, 1)
# Thresh = 0.259 , n = 1, R2 : 24.49%

# ================================================================================
# RandomizedSearchCV / XGBRegressor

# [0.3889513  0.21022229 0.47817684 0.43700818 0.34991408]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.6, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#              max_depth=4, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.03722127 0.04774503 0.05043931 0.0532322  0.05789194 0.06762408
#  0.06851707 0.0927114  0.24549156 0.27912614]
# (353, 10)
# Thresh = 0.037 , n = 10, R2 : 42.15%
# (353, 9)
# Thresh = 0.048 , n = 9, R2 : 39.54%
# (353, 8)
# Thresh = 0.050 , n = 8, R2 : 42.07%
# (353, 7)
# Thresh = 0.053 , n = 7, R2 : 37.49%
# (353, 6)
# Thresh = 0.058 , n = 6, R2 : 36.47%
# (353, 5)
# Thresh = 0.068 , n = 5, R2 : 36.11%
# (353, 4)
# Thresh = 0.069 , n = 4, R2 : 28.97%
# (353, 3)
# Thresh = 0.093 , n = 3, R2 : 21.87%
# (353, 2)
# Thresh = 0.245 , n = 2, R2 : 8.74%
# (353, 1)
# Thresh = 0.279 , n = 1, R2 : 16.13%

# ================================================================================
# GreadSearchCV / RandomizedSearchCV

# [0.41858865 0.32328823 0.30102615 0.34769842 0.22680839]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=0.6, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
#              max_depth=4, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.04261043 0.05252751 0.05453857 0.05610774 0.05743288 0.07210095
#  0.10864117 0.13237439 0.18768868 0.23597774]
# (353, 10)
# Thresh = 0.043 , n = 10, R2 : 51.19%
# (353, 9)
# Thresh = 0.053 , n = 9, R2 : 49.76%
# (353, 8)
# Thresh = 0.055 , n = 8, R2 : 46.13%
# (353, 7)
# Thresh = 0.056 , n = 7, R2 : 45.20%
# (353, 6)
# Thresh = 0.057 , n = 6, R2 : 47.86%
# (353, 5)
# Thresh = 0.072 , n = 5, R2 : 47.84%
# (353, 4)
# Thresh = 0.109 , n = 4, R2 : 41.47%
# (353, 3)
# Thresh = 0.132 , n = 3, R2 : 30.00%
# (353, 2)
# Thresh = 0.188 , n = 2, R2 : 37.78%
# (353, 1)
# Thresh = 0.236 , n = 1, R2 : 24.45%

# ================================================================================
# RandomizedSearchCV / GreadSearchCV 

# [0.36507366 0.22088026 0.50812012 0.35813739 0.36661526]
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
#              gamma=0, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.01, max_delta_step=0,
#              max_depth=4, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=300, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# [0.04101345 0.04535572 0.05355226 0.0537133  0.06017726 0.06732537
#  0.06763753 0.07481735 0.20562607 0.33078167]
# (353, 10)
# Thresh = 0.041 , n = 10, R2 : 52.31%
# (353, 9)
# Thresh = 0.045 , n = 9, R2 : 50.08%
# (353, 8)
# Thresh = 0.054 , n = 8, R2 : 46.95%
# (353, 7)
# Thresh = 0.054 , n = 7, R2 : 51.03%
# (353, 6)
# Thresh = 0.060 , n = 6, R2 : 55.40%
# (353, 5)
# Thresh = 0.067 , n = 5, R2 : 49.43%
# (353, 4)
# Thresh = 0.068 , n = 4, R2 : 49.03%
# (353, 3)
# Thresh = 0.075 , n = 3, R2 : 44.38%
# (353, 2)
# Thresh = 0.206 , n = 2, R2 : 35.99%
# (353, 1)
# Thresh = 0.331 , n = 1, R2 : 25.50%
