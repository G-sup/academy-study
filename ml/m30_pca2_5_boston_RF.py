import numpy as np
#  차원 축소
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor

datasets = load_boston()

x = datasets.data
y = datasets.target

pca =PCA(n_components=10) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)
print(x.shape) #(506, 13)
print(y.shape) #(506,)

# print(x2)
# print(x2.shape) # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # variance_ratio 변화율
# print(pca_EVR)
# print(sum(pca_EVR))

# pca = PCA()
# pca.fit(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)

# print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

# d = np.argmax(cumsum >=0.95) + 1
# print('cumsum >=0.95 : ',cumsum >=0.95)
# print('d : ',d)

# import matplotlib.pyplot as plt

# plt.plot(cumsum)
# plt.grid()
# plt.show()

#2 모델 구성

# model = RandomForestRegressor()
model = XGBRegressor(n_jobs=-1,use_label_encoder=False,eval_metric='mlogloss')

score = cross_val_score(model, x_train,y_train,cv= KFold)


print(score)

# (442, 7)
# (442,)
# [0.34307936 0.41395492 0.57884725 0.32314566 0.24108945]

# (506, 13)
# (506,)
# [0.87424423 0.7169186  0.8809538  0.8558426  0.90405688]

#  XGBRegressor
# (506, 10)
# (506,)
# [0.76064532 0.65871593 0.61308861 0.75508101 0.40586948]

# (506, 13)
# (506,)
# [0.77518526 0.89606561 0.8885825  0.90506675 0.91265505]