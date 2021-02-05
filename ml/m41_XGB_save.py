from sklearn import datasets
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,accuracy_score


#1
# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(x, y, train_size = 0.8 , random_state = 104 )



#2
model = XGBRegressor(n_estimators=1000, learning_rate = 0.01, n_jovs=8)

#3

model.fit(x_train,y_train,verbose=1,eval_metric=['rmse'], eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=20)

# fit 안에 early_stopping_rounds = 숫자

#4

aaa = model.score(x_test,y_test)

print(aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2 : ' ,r2)

result = model.evals_result()
# print(result)


# 모델 저장
import pickle
import joblib

# pickle.dump(model, open('../data/xgb_save/m_39.pickle.dat','wb'))  # dump : save랑 같다
# print('저장')

# joblib.dump(model,('../data/xgb_save/m_40.jolib.dat'))
# print('저장')

# model.save_model('../data/xgb_save/m_41.xgb.dat')
# print('저장')

# print('============================================================')
# # 모델 불러오기
# # model2 = pickle.load(open('../data/xgb_save/m_39.pickle.dat','rb'))
# model2 = joblib.load('../data/xgb_save/m_40.jolib.dat')
model2 = XGBRegressor()
model2.load_model('../data/xgb_save/m_41.xgb.dat')

print('불러오기')
r22 = model2.score(x_test,y_test)
print(r22)
