from sklearn import datasets
from xgboost import XGBClassifier,XGBRegressor, sklearn
from sklearn.datasets import load_wine
import numpy as np
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

epochs = len(result['validation_0']['merror'])
x_axis = range(0,epochs)

fig ,ax = plt.subplots()

ax.plot(x_axis,result['validation_0']['merror'],label='Train')
ax.plot(x_axis,result['validation_1']['merror'],label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')
plt.show

fig, ax =plt.subplots()

ax.plot(x_axis,result['validation_0']['mlogloss'],label='Train')
ax.plot(x_axis,result['validation_1']['mlogloss'],label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.title('XGBoost mlogloss')
plt.show()