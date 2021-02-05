from sklearn import datasets
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt

#1
# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(x, y, train_size = 0.8 , random_state = 104 )



#2
model = XGBRegressor(n_estimators=1000, learning_rate = 0.01, n_jovs=8)

#3

model.fit(x_train,y_train,verbose=1,eval_metric=['rmse','logloss'], eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=1000)

# fit 안에 early_stopping_rounds = 숫자

#4

aaa = model.score(x_test,y_test)

print(aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2 : ' ,r2)

result = model.evals_result()
print(result)

epochs = len(result['validation_0']['logloss'])
x_axis = range(0,epochs)

fig ,ax = plt.subplots()

ax.plot(x_axis,result['validation_0']['logloss'],label='Train')
ax.plot(x_axis,result['validation_1']['logloss'],label='Test')
ax.legend()
plt.ylabel('Log loss')
plt.title('XGBoost Log Loss')
plt.show

fig, ax =plt.subplots()

ax.plot(x_axis,result['validation_0']['rmse'],label='Train')
ax.plot(x_axis,result['validation_1']['rmse'],label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()