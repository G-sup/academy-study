# 25% 미만 제거

from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,plot_importance
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#1

datasets = load_iris()

x_train, x_test, y_train,y_test = train_test_split(datasets.data,datasets.target, train_size = 0.8 , random_state = 104 )
#2
# st = datetime.datetime.now()
# model  = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_jobs=-1,use_label_encoder=False)

# n_jobs = -1, 1, 4, 8 비교 8이 제일 잘나왔다 (n_jobs = : 얼마만큼의 코어를 쓰는가)
#  통상적으로 -1 이 잘나온다고 하지만 코어를 다(8) 적는게 오히려 더 잘나온다.. 

#3

model.fit(x_train,y_train,eval_metric='mlogloss')

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ',acc)
# et = datetime.datetime.now()

# print(et-st)


# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,
#             align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)

plot_importance(model) # xgboost 에 import 한다
plt.show()



# [0.01340486 0.00943773 0.24529822 0.73185919]
# acc :  1.0


# 0:00:00.095714 , 0:00:00.094801 ,-1, 8

# 0:00:00.086766  , 4

# 0:00:00.085151  , 2