from sklearn import datasets
from xgboost import XGBClassifier,XGBRegressor, sklearn
from sklearn.datasets import load_wine
import numpy as np
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split


#1
# x, y = load_boston(return_X_y=True)
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=104)



#2
model = XGBClassifier(n_estimators=500, learning_rate = 0.01, n_jovs=8)

#3
model.fit(x_train,y_train,verbose=1,eval_metric=['merror','mlogloss'], eval_set=[(x_train,y_train),(x_test,y_test)])

aaa = model.score(x_test,y_test)

print(aaa)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('acc : ' ,acc)

result = model.evals_result()
# print(result)