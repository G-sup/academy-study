
#  차원 축소
import numpy
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target



pca =PCA(n_components=15) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)

print(x.shape) #(442, 10)
print(y.shape) #(442,)

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

# (569, 15)
# (569,)
# [0.91208791 0.94505495 0.95604396 0.91208791 0.97802198]

# (569, 30)
# (569,)
# [0.95604396 0.94505495 0.98901099 0.94505495 0.96703297]

#  XGBRegressor
# (569, 15)
# (569,)
# [0.97802198 0.96703297 0.97802198 0.93406593 0.91208791]

# (569, 30)
# (569,)
# [0.95604396 0.93406593 0.93406593 0.97802198 0.95604396]