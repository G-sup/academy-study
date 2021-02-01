from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datasets = load_wine()

x_train, x_test, y_train,y_test = train_test_split(datasets.data, datasets.target, train_size = 0.8 , random_state = 104 )


#2

model  = GradientBoostingClassifier(max_depth=4)

#3

model.fit(x_train,y_train)

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ',acc)

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [0.26835358 0.03211674 0.00308965 0.00127517 0.01008336 0.00080833
#  0.1475908  0.00380414 0.00200624 0.04918747 0.00333934 0.22583462
#  0.25251058]
# acc :  0.8611111111111112