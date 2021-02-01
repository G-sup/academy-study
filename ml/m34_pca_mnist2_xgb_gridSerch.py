import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np 
from numpy.core.fromnumeric import cumsum, shape
from sklearn import datasets 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
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

parameters = [
    {'n_estimators':[100,200,300], 'learning_rate':[0.1,0.3,0.001,0.01], 'max_depth':[4,5,6]},
    {'n_estimators':[90,100,110], 'learning_rate':[0.1,0.001,0.01], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators':[100,110], 'learning_rate':[0.1,0.5,0.001], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.9]}
]

#2 모델 구성

model = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), parameters, cv = KFold  , verbose=1) 

score = cross_val_score(model, x_train,y_train,cv= KFold)


print(score)
