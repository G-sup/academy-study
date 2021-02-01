
#  차원 축소
import numpy
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier

datasets = load_wine()
x = datasets.data
y = datasets.target

pca =PCA(n_components=7) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)

print(x.shape) #(178, 13)
print(y.shape) #(178,)

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

# print(x2)
# print(x2.shape) # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # variance_ratio 변화율
# print(pca_EVR)
# print(sum(pca_EVR))

# pca = PCA()
# pca.fit(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)

# print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

# d = np.argmax(cumsum >=0.95) + 1 # cumsum 0.95 이상 부터 True
# print('cumsum >=0.95 : ',cumsum >=0.95)
# print('d : ',d)

# import matplotlib.pyplot as plt

# plt.plot(cumsum)
# plt.grid()
# plt.show()

#2 모델 구성

# model = RandomForestClassifier()
model = XGBClassifier(n_jobs=-1,use_label_encoder=False,eval_metric='mlogloss')

score = cross_val_score(model, x_train,y_train,cv= KFold)


print(score)

# (178, 7)
# (178,)
# [0.89655172 0.89655172 1.         0.96428571 0.92857143]

# (178, 13)
# (178,)
# [1. 1. 1. 1. 1.]

#  XGBRegressor
# (178, 7)
# (178,)
# [0.89655172 0.93103448 0.92857143 0.89285714 0.89285714]

# (178, 13)
# (178,)
# [1.         0.93103448 0.96428571 0.96428571 1.        ]