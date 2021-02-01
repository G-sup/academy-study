import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA # decomposition 분해 
from xgboost import XGBClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
y = np.append(y_train,y_test, axis=0)

print(x.shape)
x = x.reshape(70000,-1)
# 실습 
# pca로 0.95 이상 개수 
# pca 사용해서 확인

# pca =PCA(n_components=154) # n_components = : 컬럼을 몇개로 줄이는지

# x = pca.fit_transform(x)

# x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

# KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)


# ============================================
# pca =PCA(n_components=154) 

# x2 = pca.fit_transform(x)

# print(x2)
# print(x2.shape) # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # variance_ratio 변화율
# print(pca_EVR)
# print(sum(pca_EVR))
# =====================0.95 이상 확인용====================================

pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)

print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

d = np.argmax(cumsum >=0.95) + 1
print('cumsum >=0.95 : ',cumsum >=0.95)
print('d : ',d)


# =======================1.0이상 확인용=====================================
# pca = PCA()
# pca.fit(x)

# cumsum = np.cumsum(pca.explained_variance_ratio_)

# print('cumsum : ', cumsum) #  cumsum : 결과 값의 변수 type을 설정하면서 누적 sum을 함. 통상적으로 0.95 이상이면 비슷하다고 한다

# d = np.argmax(cumsum >=1.0) + 1
# print('cumsum >=1.0 : ',cumsum >=1.0)
# print('d : ',d)
# ==========================================================================

# import matplotlib.pyplot as plt

# plt.plot(cumsum)
# plt.grid()
# plt.show()

# 3 모델
# model = XGBClassifier(n_jobs=-1,use_label_encoder=False,eval_metric='mlogloss')

# score = cross_val_score(model, x_train,y_train,cv= KFold)


# print(score)