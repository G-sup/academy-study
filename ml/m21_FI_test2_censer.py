from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#1

datasets = load_breast_cancer()

x_train, x_test, y_train,y_test = train_test_split(datasets.data, datasets.target, train_size = 0.8 , random_state = 104 )

#2

model  = DecisionTreeClassifier(max_depth=4)

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

df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,4),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,4))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

model = DecisionTreeClassifier()

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)


# [0.         0.03371907 0.         0.         0.         0.
#  0.         0.02811128 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.00681367 0.         0.78743833 0.
#  0.         0.         0.01332894 0.12056797 0.01002074 0.        ]
# acc :  0.9122807017543859

# [0.78743833 0.12056797 0.01332894 0.         0.01002074 0.02811128
#  0.04053275]
# acc :  0.9210526315789473