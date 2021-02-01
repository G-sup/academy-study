
#  차원 축소
import numpy
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target


pca =PCA(n_components=2) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)

print(x.shape) #(150, 4)
print(y.shape) #(150,)

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

# print(x2)
# print(x2.shape) # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # variance_ratio 변화율
# print(pca_EVR)
# print(sum(pca_EVR))


# 6개 0.8942875900367643
# 7개 0.9479436357350414
# 8개 0.9913119559917797
# 9개 0.9991439470098977


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

# (150, 2)
# (150,)
# [0.91666667 0.95833333 1.         0.875      0.91666667]

# (150, 4)
# (150,)
# [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333]

#  XGBRegressor
# (150, 2)
# (150,)
# [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333]

# (150, 4)
# (150,)
# [0.95833333 0.95833333 0.91666667 0.91666667 0.95833333]