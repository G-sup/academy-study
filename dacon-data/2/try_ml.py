import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Conv2D, Conv1D ,Flatten, MaxPooling2D,MaxPool1D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import dropout
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score,KFold,RandomizedSearchCV
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

#1
train = pd.read_csv('./dacon-data/2/train.csv', index_col=[0,2], header=0) 
pred = pd.read_csv('./dacon-data/2/test.csv', index_col=[0,1], header=0) 
sub = pd.read_csv('./dacon-data/2/submission.csv', index_col=0, header=0) 

y = train['digit']
x = train.drop(['digit'],1)
print(x)
print(y)
x_pred = pred.values

# pca =PCA(n_components=99) # n_components = : 컬럼을 몇개로 줄이는지

# x2 = pca.fit_transform(x)

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

pca =PCA(n_components=100) # n_components = : 컬럼을 몇개로 줄이는지

x = pca.fit_transform(x)
x_pred = pca.fit_transform(x_pred)
# print(x_pred.shape)
# print(y.shape)
# print(x.shape)

x_train , x_test,y_train ,y_test = train_test_split( x, y, train_size = 0.8, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (shuffle=False : 순차적)

# parameters = [
#     {'n_estimators':[100,200,300], 'learning_rate':[0.1,0.3,0.001,0.01], 'max_depth':[4,5,6]},
#     {'n_estimators':[90,100,110], 'learning_rate':[0.1,0.001,0.01], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
#     {'n_estimators':[100,110], 'learning_rate':[0.1,0.5,0.001], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.9]}
# ]

#2 모델 구성

# model = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss', eval_set=[(x_train,y_train),(x_test,y_test)]), parameters, cv = KFold , verbose=True)# mlogloss = loss
# score = cross_val_score(model, x_train,y_train,cv= KFold)

model = XGBClassifier(n_jobs=8,use_label_encoder=False,eval_metric='mlogloss')
model.fit(x_train,y_train,eval_metric='mlogloss', eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=50)

#4

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ',acc)
y_pred = model.predict(x_pred)
# print (y_pred)
# print(score)

# acc = model.score(x_test,y_test)

# y_pred=pd.DataFrame(y_pred)
# file_path='./dacon-data/2/result/result_ml_'+'.csv'
# y_pred.to_csv(file_path)
