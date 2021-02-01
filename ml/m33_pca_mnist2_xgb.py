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

pca =PCA(n_components=713) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

# 3 모델
model = XGBClassifier(n_jobs=-1,use_label_encoder=False,eval_metric='mlogloss')

model.fit(x_train,y_train,eval_metric='mlogloss')

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ',acc)