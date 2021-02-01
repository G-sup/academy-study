from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datasets = load_boston()

x_train, x_test, y_train,y_test = train_test_split(datasets.data, datasets.target, train_size = 0.8 , random_state = 104 )


#2

# model  = GradientBoostingRegressor(max_depth=4)
model = XGBRegressor(n_jobs=-1,use_label_encoder=False)

#3

model.fit(x_train,y_train,eval_metric='mlogloss')

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('r2 : ',acc)

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [2.43095171e-02 4.65441455e-04 4.11455484e-03 2.93420408e-04
#  2.45750255e-02 2.86885688e-01 7.10021741e-03 6.24000775e-02
#  5.60753521e-03 1.61305776e-02 2.74419754e-02 1.04731663e-02
#  5.30202803e-01]
# r2 :  0.9153255256085908