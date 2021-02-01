# 0인 컬럼 제거

from sys import platform
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#1

datasets = load_iris()

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
    n_features = datasets.data.shape[-1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [0.01699235 0.         0.04451513 0.93849252]
# 1.0

# [0.01699235 0.04451513 0.93849252]
# acc :  1.0


fi = model.feature_importances_
new_data = []
feature = []
for i in range(len(fi)):
    if fi[i] != 0:
        new_data.append(datasets.data[:,i])
        feature.append(datasets.feature_names[i])

new_data = np.array(new_data)
new_data = np.transpose(new_data)

x_train2,x_test2,y_train2,y_test2 = train_test_split(new_data,datasets.target,train_size = 0.8, random_state = 33)

#2. 모델
model2 = DecisionTreeClassifier(max_depth = 4)

#3. 훈련
model2.fit(x_train2, y_train2)

#4. 평가 예측
acc2 = model2.score(x_test2,y_test2)

print('acc 칼럼 지우고!!!! : ', acc2)

####### dataset >> new_data 로 바꾸고 featurename 부분을 feature 리스트로 바꿔줌!!!
def plot_feature_importances_dataset(model):
    n_features = new_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), feature)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model2)
plt.show()