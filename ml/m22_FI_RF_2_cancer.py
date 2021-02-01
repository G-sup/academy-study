from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1

datasets = load_breast_cancer()

x_train, x_test, y_train,y_test = train_test_split(datasets.data, datasets.target, train_size = 0.8 , random_state = 104 )


#2

model  = RandomForestClassifier(max_depth=4)

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

def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result



# [0.04548679 0.01211865 0.04632223 0.03824626 0.00317346 0.01036818
#  0.05698576 0.10402952 0.00293986 0.00206411 0.02802977 0.00363749
#  0.0175927  0.03327623 0.0035293  0.00215433 0.00649654 0.00292257
#  0.00325811 0.0020514  0.10032757 0.01274355 0.13711278 0.12018144
#  0.00542521 0.01461017 0.04825805 0.12216884 0.00877435 0.0057148 ]
# acc :  0.9473684210526315