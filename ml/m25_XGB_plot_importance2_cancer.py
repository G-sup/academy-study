from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,plot_importance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datasets = load_breast_cancer()

x_train, x_test, y_train,y_test = train_test_split(datasets.data, datasets.target, train_size = 0.8 , random_state = 104 )

#2

# model  = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_jobs=-1,use_label_encoder=False)

#3

model.fit(x_train,y_train,eval_metric='mlogloss')

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ',acc)


'''
def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
'''

plot_importance(model) # xgboost 에 import 한다
plt.show()


# [6.14643107e-05 1.23920937e-02 3.98779266e-04 3.58328318e-03
#  9.70612664e-05 2.46249537e-03 6.88332493e-04 7.39694441e-02
#  4.04907237e-03 2.60835854e-03 1.56303069e-03 2.96224643e-03
#  5.61403308e-03 9.02763048e-04 1.86106232e-04 1.64839590e-03
#  1.12392160e-03 6.95009960e-04 6.99047980e-04 1.82984452e-02
#  6.13011146e-03 4.70405339e-02 7.17136596e-01 1.14237330e-02
#  7.75335439e-04 7.16154585e-04 1.20525484e-02 6.88148511e-02
#  2.94428951e-04 1.61232225e-03]
# acc :  0.9210526315789473